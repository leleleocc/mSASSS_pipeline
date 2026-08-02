#!/usr/bin/env python3
"""Train EfficientNet-B0 to predict ordinal up/down scores for each VU."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.stage2.data import (
    AugmentationConfig,
    VUSample,
    ZhongriVUDataset,
    assign_patient_folds,
    build_balanced_sampler,
    load_zhongri_samples,
)
from src.stage2.model import (
    VUOrdinalEfficientNet,
    load_stage1_backbone,
    ordinal_loss,
    trainable_parameter_counts,
)


STAGE2_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STAGE2_ROOT.parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "raw_data/zhongri/1-after-trim"
DEFAULT_PROJECT = STAGE2_ROOT / "outputs/efficientnet"
MPL_CONFIG = STAGE2_ROOT / ".matplotlib"
MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--fold", type=int, default=0, help="validation fold index")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0", help="one CUDA index or cpu")
    parser.add_argument("--backbone-lr", type=float, default=3e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--sampler-power", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument(
        "--stage1-weights",
        type=Path,
        default=None,
        help="optional leak-safe Stage-1 EfficientNet checkpoint; only its backbone is loaded",
    )
    parser.add_argument("--no-imagenet", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", default=None, help="defaults to fold_N")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--exist-ok", action="store_true")

    parser.add_argument("--rotation", type=float, default=3.0)
    parser.add_argument("--translation", type=float, default=0.04)
    parser.add_argument("--field-scale", type=float, default=0.10)
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--contrast", type=float, default=0.10)
    parser.add_argument("--brightness", type=float, default=0.03)
    parser.add_argument("--noise-probability", type=float, default=0.25)
    parser.add_argument("--noise-sigma", type=float, default=0.008)
    parser.add_argument("--blur-probability", type=float, default=0.15)
    parser.add_argument("--blur-sigma-max", type=float, default=0.8)
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value.casefold() == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; pass --device cpu")
    try:
        index = int(value)
    except ValueError as exc:
        raise RuntimeError("--device must be one CUDA index or cpu") from exc
    if index < 0 or index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {index} does not exist")
    return torch.device(f"cuda:{index}")


def init_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 1:
        return False, 0, 0, 1
    if not torch.cuda.is_available():
        raise RuntimeError("distributed training requires CUDA")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return True, rank, local_rank, world_size


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_scheduler(
    optimizer: torch.optim.Optimizer, epochs: int, warmup_epochs: int
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_epochs = min(max(warmup_epochs, 0), max(epochs - 1, 0))

    def multiplier(epoch: int) -> float:
        if warmup_epochs and epoch < warmup_epochs:
            return 0.1 + 0.9 * (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs - 1, 1)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


def freeze_batch_norm_stats(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            child.eval()


def set_backbone_trainable(model: VUOrdinalEfficientNet, trainable: bool) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(trainable)


def grade_counts(samples: Sequence[VUSample]) -> dict[str, dict[str, int]]:
    return {
        endpoint: {
            str(grade): int(Counter(getattr(sample, f"{endpoint}_score") for sample in samples).get(grade, 0))
            for grade in range(4)
        }
        for endpoint in ("up", "down")
    }


def quadratic_weighted_kappa(confusion: np.ndarray) -> float:
    confusion = confusion.astype(np.float64)
    count = confusion.sum()
    if count == 0:
        return float("nan")
    observed = confusion / count
    expected = np.outer(confusion.sum(axis=1), confusion.sum(axis=0)) / (count * count)
    indices = np.arange(4, dtype=np.float64)
    weights = ((indices[:, None] - indices[None, :]) / 3.0) ** 2
    denominator = float((weights * expected).sum())
    return 1.0 if denominator == 0 else float(1.0 - (weights * observed).sum() / denominator)


@torch.no_grad()
def evaluate(
    model: VUOrdinalEfficientNet,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[dict[str, Any], np.ndarray]:
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        ordinal = batch["ordinal_targets"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            outputs = model(images)
            loss, _ = ordinal_loss(outputs["ordinal_logits"], ordinal)
        batch_size = images.shape[0]
        loss_sum += float(loss) * batch_size
        sample_count += batch_size
        predictions.append(outputs["scores"].cpu().numpy())
        targets.append(batch["scores"].numpy())

    predicted = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets, axis=0)
    absolute = np.abs(predicted - target)
    confusion = np.zeros((2, 4, 4), dtype=np.int64)
    for endpoint in range(2):
        np.add.at(confusion[endpoint], (target[:, endpoint], predicted[:, endpoint]), 1)
    metrics: dict[str, Any] = {
        "val_loss": loss_sum / max(sample_count, 1),
        "mean_mae": float(absolute.mean()),
        "up_mae": float(absolute[:, 0].mean()),
        "down_mae": float(absolute[:, 1].mean()),
        "mean_exact_accuracy": float((predicted == target).mean()),
        "up_exact_accuracy": float((predicted[:, 0] == target[:, 0]).mean()),
        "down_exact_accuracy": float((predicted[:, 1] == target[:, 1]).mean()),
        "mean_within_one": float((absolute <= 1).mean()),
        "up_qwk": quadratic_weighted_kappa(confusion[0]),
        "down_qwk": quadratic_weighted_kappa(confusion[1]),
    }
    metrics["mean_qwk"] = float((metrics["up_qwk"] + metrics["down_qwk"]) / 2)
    return metrics, confusion


def save_checkpoint(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def plot_history(csv_path: Path, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    if not rows:
        return
    epochs = [int(row["epoch"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, [float(row["train_loss"]) for row in rows], label="train")
    axes[0].plot(epochs, [float(row["val_loss"]) for row in rows], label="validation")
    axes[0].set(xlabel="epoch", ylabel="ordinal BCE", title="Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(epochs, [float(row["mean_mae"]) for row in rows], label="mean MAE")
    axes[1].plot(epochs, [float(row["mean_qwk"]) for row in rows], label="mean QWK")
    axes[1].set(xlabel="epoch", title="Validation scoring metrics")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def main() -> int:
    args = parse_args()
    if args.folds < 2 or not 0 <= args.fold < args.folds:
        raise ValueError("--fold must be in [0, --folds)")
    if args.epochs < 1 or args.batch < 1 or args.workers < 0 or args.crop_size < 32:
        raise ValueError("invalid epochs, batch, workers, or crop size")
    if args.patience < 0 or args.freeze_backbone_epochs < 0:
        raise ValueError("patience and freeze-backbone-epochs cannot be negative")
    if not 0 <= args.sampler_power <= 1:
        raise ValueError("sampler-power must be in [0, 1]")
    if args.resume is not None and args.stage1_weights is not None:
        raise ValueError("--resume and --stage1-weights are mutually exclusive")

    augmentation = AugmentationConfig(
        rotation_deg=args.rotation,
        translation_fraction=args.translation,
        field_scale=args.field_scale,
        gamma=args.gamma,
        contrast=args.contrast,
        brightness=args.brightness,
        noise_probability=args.noise_probability,
        noise_sigma=args.noise_sigma,
        blur_probability=args.blur_probability,
        blur_sigma_max=args.blur_sigma_max,
    )
    augmentation.validate()
    distributed, rank, local_rank, world_size = init_distributed()
    device = torch.device(f"cuda:{local_rank}") if distributed else resolve_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    seed_everything(args.seed + rank)

    samples = load_zhongri_samples(args.source)
    assignments = assign_patient_folds(samples, args.folds, args.seed)
    train_samples = [sample for sample in samples if assignments[sample.patient_id] != args.fold]
    val_samples = [sample for sample in samples if assignments[sample.patient_id] == args.fold]
    train_patients = sorted({sample.patient_id for sample in train_samples})
    val_patients = sorted({sample.patient_id for sample in val_samples})
    if set(train_patients) & set(val_patients):
        raise RuntimeError("patient leakage between train and validation splits")

    name = args.name or f"fold_{args.fold}"
    output_dir = (args.project.expanduser() / name).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and args.resume is None and not args.exist_ok:
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; choose --name or pass --exist-ok"
        )
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "results.csv"

    split_summary = {
        "fold": args.fold,
        "folds": args.folds,
        "seed": args.seed,
        "train": {
            "patients": train_patients,
            "patient_count": len(train_patients),
            "vus": len(train_samples),
            "grades": grade_counts(train_samples),
        },
        "validation": {
            "patients": val_patients,
            "patient_count": len(val_patients),
            "vus": len(val_samples),
            "grades": grade_counts(val_samples),
        },
    }
    if rank == 0:
        (output_dir / "split.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")
        (output_dir / "args.json").write_text(json.dumps(serializable_args(args), indent=2), encoding="utf-8")
    if distributed:
        dist.barrier()

    train_dataset = ZhongriVUDataset(
        train_samples,
        augment=True,
        crop_size=args.crop_size,
        augmentation=augmentation,
        seed=args.seed,
    )
    val_dataset = ZhongriVUDataset(
        val_samples,
        augment=False,
        crop_size=args.crop_size,
        augmentation=augmentation,
        seed=args.seed,
    )
    sampler = build_balanced_sampler(train_samples, args.sampler_power, args.seed) if not distributed else None
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = {
        "batch_size": args.batch,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "persistent_workers": args.workers > 0,
    }
    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(train_dataset, shuffle=False, sampler=train_sampler, **loader_options)
    else:
        train_loader = DataLoader(
            train_dataset,
            shuffle=sampler is None,
            sampler=sampler,
            generator=generator if sampler is None else None,
            **loader_options,
        )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    pretrained = not args.no_imagenet and args.stage1_weights is None and args.resume is None
    try:
        base_model = VUOrdinalEfficientNet(
            pretrained=pretrained,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )
    except Exception as exc:
        if not pretrained:
            raise
        raise RuntimeError(
            "failed to load/download ImageNet EfficientNet-B0 weights; fix access or pass --no-imagenet"
        ) from exc
    initialization: dict[str, Any] = {"type": "imagenet" if pretrained else "random"}
    if args.stage1_weights is not None:
        initialization = {"type": "stage1_backbone", **load_stage1_backbone(base_model, args.stage1_weights)}
    base_model.to(device)
    model: nn.Module = base_model
    if distributed:
        model = DDP(base_model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    optimizer = torch.optim.AdamW(
        [
            {"params": base_model.backbone.parameters(), "lr": args.backbone_lr},
            {
                "params": [
                    parameter
                    for name_, parameter in base_model.named_parameters()
                    if not name_.startswith("backbone.")
                ],
                "lr": args.head_lr,
            },
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = make_scheduler(optimizer, args.epochs, args.warmup_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 0
    best_mae = math.inf
    best_val_loss = math.inf
    best_epoch = 0
    patience_count = 0
    if args.resume is not None:
        resume_path = args.resume.expanduser().resolve()
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        base_model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint.get("scaler_state", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        best_mae = float(checkpoint.get("best_mae", math.inf))
        best_val_loss = float(checkpoint.get("best_val_loss", math.inf))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        patience_count = int(checkpoint.get("patience_count", 0))
        initialization = checkpoint.get("initialization", {"type": "resume"})

    fields = [
        "epoch",
        "train_loss",
        "train_up_loss",
        "train_down_loss",
        "val_loss",
        "mean_mae",
        "up_mae",
        "down_mae",
        "mean_exact_accuracy",
        "up_exact_accuracy",
        "down_exact_accuracy",
        "mean_within_one",
        "mean_qwk",
        "up_qwk",
        "down_qwk",
        "backbone_lr",
        "head_lr",
        "backbone_trainable",
        "seconds",
    ]
    if rank == 0:
        if start_epoch == 0 or not history_path.exists():
            with history_path.open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=fields).writeheader()
    if distributed:
        dist.barrier()

    counts = trainable_parameter_counts(base_model)
    if rank == 0:
        print(
            f"Stage-2 EfficientNet-B0 fold {args.fold}/{args.folds - 1}: "
            f"{len(train_patients)} patients/{len(train_samples)} VUs train, "
            f"{len(val_patients)} patients/{len(val_samples)} VUs val, "
            f"input={args.crop_size}x{args.crop_size}, device={device}, AMP={amp_enabled}"
        )
        print(f"Parameters: total={counts['total']:,}, backbone={counts['backbone']:,}, head={counts['head']:,}")
        print(f"Initialization: {initialization}")

    last_metrics: dict[str, Any] = {}
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        started = time.perf_counter()
        backbone_trainable = epoch >= args.freeze_backbone_epochs
        set_backbone_trainable(base_model, backbone_trainable)
        model.train()
        freeze_batch_norm_stats(base_model.backbone)
        if not backbone_trainable:
            model.backbone.eval()

        train_sums = {"loss": 0.0, "up_loss": 0.0, "down_loss": 0.0}
        seen = 0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            ordinal = batch["ordinal_targets"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model(images)
                loss, parts = ordinal_loss(outputs["ordinal_logits"], ordinal)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch + 1}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            batch_size = images.shape[0]
            seen += batch_size
            train_sums["loss"] += float(loss.detach()) * batch_size
            train_sums["up_loss"] += float(parts["up_loss"].detach()) * batch_size
            train_sums["down_loss"] += float(parts["down_loss"].detach()) * batch_size

        metrics, confusion = evaluate(base_model, val_loader, device, amp_enabled)
        last_metrics = metrics
        monitor = (float(metrics["mean_mae"]), float(metrics["val_loss"]))
        improved = monitor < (best_mae, best_val_loss)
        if improved:
            best_mae, best_val_loss = monitor
            best_epoch = epoch + 1
            patience_count = 0
        else:
            patience_count += 1
        duration = time.perf_counter() - started
        row = {
            "epoch": epoch + 1,
            "train_loss": train_sums["loss"] / seen,
            "train_up_loss": train_sums["up_loss"] / seen,
            "train_down_loss": train_sums["down_loss"] / seen,
            **{key: value for key, value in metrics.items() if not isinstance(value, (list, dict))},
            "backbone_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]["lr"],
            "backbone_trainable": int(backbone_trainable),
            "seconds": duration,
        }
        if rank == 0:
            with history_path.open("a", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=fields).writerow(row)

        scheduler.step()
        if rank == 0:
            payload = {
                "model_state": base_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "epoch": epoch,
                "best_mae": best_mae,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "patience_count": patience_count,
                "metrics": metrics,
                "confusion_matrices": confusion.tolist(),
                "args": serializable_args(args),
                "initialization": initialization,
                "parameter_counts": counts,
                "split": split_summary,
            }
            save_checkpoint(payload, weights_dir / "last.pt")
            if improved:
                best_payload = {
                    key: value
                    for key, value in payload.items()
                    if key not in {"optimizer_state", "scheduler_state", "scaler_state"}
                }
                save_checkpoint(best_payload, weights_dir / "best.pt")
            plot_history(history_path, output_dir / "results.png")
            print(
                f"epoch {epoch + 1:03d}/{args.epochs} "
                f"loss={row['train_loss']:.4f} val={metrics['val_loss']:.4f} "
                f"MAE={metrics['mean_mae']:.3f} exact={metrics['mean_exact_accuracy']:.3f} "
                f"QWK={metrics['mean_qwk']:.3f} {'BEST' if improved else ''}"
            )
        if distributed:
            dist.barrier()
        if args.patience and patience_count >= args.patience:
            if rank == 0:
                print(f"Early stopping after {args.patience} epochs without MAE improvement")
            break

    if rank == 0:
        summary = {
            "best_epoch": best_epoch,
            "best_mean_mae": best_mae,
            "best_val_loss": best_val_loss,
            "last_metrics": last_metrics,
            "weights": str(weights_dir / "best.pt"),
            "initialization": initialization,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Best epoch: {best_epoch}, mean MAE: {best_mae:.4f}")
        print(f"Results saved to {output_dir}")
    cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
