#!/usr/bin/env python3
"""Train EfficientNet-B0 to regress one ordered set of 19 spine keypoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.stage1.efficientnet import (
    EfficientNetKeypointModel,
    SpineKeypointDataset,
    evaluate_model,
    keypoint_loss,
    normalize_image_size,
    seed_worker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent / "data/spine_keypoints_19pt/data.yaml",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=[1024, 768],
        metavar="N",
        help="one square size, or height width (default: 1024 768)",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0", help="CUDA index such as 0, or cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--coordinate-gain", type=float, default=10.0)
    parser.add_argument("--structure-gain", type=float, default=2.0)
    parser.add_argument("--class-gain", type=float, default=0.1)
    parser.add_argument("--degrees", type=float, default=5.0)
    parser.add_argument("--translate", type=float, default=0.03)
    parser.add_argument("--scale", type=float, default=0.10)
    parser.add_argument(
        "--project",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs/efficientnet",
    )
    parser.add_argument("--name", default="efficientnet_b0_19pt")
    parser.add_argument("--resume", type=Path, default=None, help="resume from a last.pt checkpoint")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value.casefold() == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false; pass --device cpu")
    try:
        index = int(value.split(",", maxsplit=1)[0])
    except ValueError as exc:
        raise RuntimeError("--device must be one CUDA index (for example 0) or cpu") from exc
    if index < 0 or index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {index} does not exist; found {torch.cuda.device_count()} devices")
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


def freeze_batch_norm_stats(model: nn.Module) -> None:
    """Keep ImageNet BatchNorm statistics fixed for this very small dataset."""
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


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


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_score: float,
    best_epoch: int,
    patience_count: int,
    metrics: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "model_state": unwrap_model(model).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": epoch,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "patience_count": patience_count,
        "metrics": metrics,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }


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
    axes[0].plot(epochs, [float(row["train_total_loss"]) for row in rows], label="total")
    axes[0].plot(epochs, [float(row["train_heatmap_loss"]) for row in rows], label="heatmap")
    axes[0].plot(epochs, [float(row["train_coordinate_loss"]) for row in rows], label="coordinate")
    if "train_structure_loss" in rows[0]:
        axes[0].plot(
            epochs,
            [float(row["train_structure_loss"]) for row in rows],
            label="structure",
        )
    axes[0].set(xlabel="epoch", ylabel="loss", title="Training losses")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        epochs,
        [float(row["mean_error_image_diag_pct"]) for row in rows],
        label="normalized MRE",
    )
    axes[1].plot(epochs, [float(row["mean_error_px"]) for row in rows], label="MRE (px)")
    axes[1].set(xlabel="epoch", ylabel="validation error", title="Validation keypoint error")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    image_size = normalize_image_size(args.imgsz)
    if any(value < 64 or value % 32 for value in image_size):
        raise RuntimeError("each --imgsz dimension must be at least 64 and divisible by 32")
    if args.epochs < 1 or args.batch < 1 or args.workers < 0:
        raise RuntimeError("epochs and batch must be positive; workers cannot be negative")
    data_yaml = args.data.resolve()
    if not data_yaml.is_file():
        raise RuntimeError(f"dataset YAML does not exist: {data_yaml}")
    output_dir = (args.project / args.name).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and args.resume is None and not args.exist_ok:
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; choose another --name or pass --exist-ok"
        )
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "results.csv"
    distributed, rank, local_rank, world_size = init_distributed()
    device = torch.device(f"cuda:{local_rank}") if distributed else resolve_device(args.device)
    amp_enabled = device.type == "cuda" and not args.no_amp
    seed_everything(args.seed + rank)

    train_dataset = SpineKeypointDataset(
        data_yaml,
        args.train_split,
        args.imgsz,
        augment=True,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
    )
    val_dataset = SpineKeypointDataset(data_yaml, args.val_split, args.imgsz, augment=False)
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
        train_loader = DataLoader(train_dataset, sampler=train_sampler, shuffle=False, **loader_options)
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    try:
        base_model = EfficientNetKeypointModel(pretrained=not args.no_pretrained).to(device)
    except Exception as exc:
        if args.no_pretrained:
            raise
        raise RuntimeError(
            "failed to load/download the torchvision EfficientNet-B0 ImageNet weights; "
            "fix network access or use --no-pretrained for a scratch run"
        ) from exc
    model: nn.Module = base_model
    if distributed:
        model = DDP(base_model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    head_parameters = list(base_model.lat4.parameters()) + list(base_model.lat8.parameters())
    head_parameters += list(base_model.lat16.parameters()) + list(base_model.lat32.parameters())
    head_parameters += list(base_model.smooth16.parameters()) + list(base_model.smooth8.parameters())
    head_parameters += list(base_model.smooth4.parameters()) + list(base_model.heatmap_head.parameters())
    head_parameters += list(base_model.classifier.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": base_model.backbone.parameters(), "lr": args.backbone_lr},
            {"params": head_parameters, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = make_scheduler(optimizer, args.epochs, args.warmup_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 0
    best_score = math.inf
    best_epoch = 0
    patience_count = 0
    if args.resume is not None:
        resume_path = args.resume.resolve()
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        base_model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint.get("scaler_state", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint.get("best_score", math.inf))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        patience_count = int(checkpoint.get("patience_count", 0))

    fieldnames = [
        "epoch",
        "train_total_loss",
        "train_heatmap_loss",
        "train_coordinate_loss",
        "train_structure_loss",
        "train_adjacent_loss",
        "train_pair_loss",
        "train_order_loss",
        "train_class_loss",
        "mean_error_px",
        "median_error_px",
        "p95_error_px",
        "mean_error_image_diag_pct",
        "pck_0.5pct_image_diag",
        "pck_1pct_image_diag",
        "pck_2pct_image_diag",
        "class_accuracy",
        "backbone_lr",
        "head_lr",
        "seconds",
    ]
    if rank == 0:
        if start_epoch == 0 or not history_path.exists():
            with history_path.open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=fieldnames).writeheader()
        (output_dir / "args.json").write_text(
            json.dumps(
                {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"Training EfficientNet-B0: {len(train_dataset)} train, {len(val_dataset)} val, "
            f"device={device}, imgsz={image_size[0]}x{image_size[1]} (HxW), "
            f"batch={args.batch}, AMP={amp_enabled}"
        )
    if distributed:
        dist.barrier()
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        started = time.perf_counter()
        model.train()
        freeze_batch_norm_stats(base_model)
        sums = {
            "total_loss": 0.0,
            "heatmap_loss": 0.0,
            "coordinate_loss": 0.0,
            "structure_loss": 0.0,
            "adjacent_loss": 0.0,
            "pair_loss": 0.0,
            "order_loss": 0.0,
            "class_loss": 0.0,
        }
        seen = 0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            class_ids = batch["class_id"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model(images)
                loss, parts = keypoint_loss(
                    outputs,
                    targets,
                    class_ids,
                    sigma=args.sigma,
                    coordinate_gain=args.coordinate_gain,
                    structure_gain=args.structure_gain,
                    class_gain=args.class_gain,
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch + 1}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            batch_size = images.shape[0]
            seen += batch_size
            for key in sums:
                sums[key] += parts[key] * batch_size

        scheduler.step()
        if distributed:
            dist.barrier()

        should_stop = False
        if rank == 0:
            metrics, _ = evaluate_model(base_model, val_loader, device, args.imgsz, amp=amp_enabled)
            duration = time.perf_counter() - started
            score = metrics["mean_error_image_diag_pct"]
            improved = score < best_score
            if improved:
                best_score = score
                best_epoch = epoch + 1
                patience_count = 0
            else:
                patience_count += 1
            row: dict[str, Any] = {
                "epoch": epoch + 1,
                "train_total_loss": sums["total_loss"] / seen,
                "train_heatmap_loss": sums["heatmap_loss"] / seen,
                "train_coordinate_loss": sums["coordinate_loss"] / seen,
                "train_structure_loss": sums["structure_loss"] / seen,
                "train_adjacent_loss": sums["adjacent_loss"] / seen,
                "train_pair_loss": sums["pair_loss"] / seen,
                "train_order_loss": sums["order_loss"] / seen,
                "train_class_loss": sums["class_loss"] / seen,
                **metrics,
                "backbone_lr": optimizer.param_groups[0]["lr"],
                "head_lr": optimizer.param_groups[1]["lr"],
                "seconds": duration,
            }
            with history_path.open("a", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=fieldnames).writerow(row)
            payload = checkpoint_payload(
                base_model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_score,
                best_epoch,
                patience_count,
                metrics,
                args,
            )
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
                f"loss={row['train_total_loss']:.4f} MRE={metrics['mean_error_px']:.2f}px "
                f"diag={score:.3f}% PCK@1%={metrics['pck_1pct_image_diag']:.3f} "
                f"class={metrics['class_accuracy']:.3f} {'BEST' if improved else ''}"
            )
            should_stop = args.patience > 0 and patience_count >= args.patience
            if should_stop:
                print(
                    f"Early stopping: no normalized-MRE improvement for {args.patience} epochs; "
                    f"best={best_score:.4f}%"
                )

        if distributed:
            stop_tensor = torch.tensor(int(should_stop), device=device)
            dist.broadcast(stop_tensor, src=0)
            should_stop = bool(stop_tensor.item())
        if should_stop:
            break

    if rank == 0:
        print(f"Best epoch: {best_epoch}, normalized MRE: {best_score:.4f}%")
        print(f"Results saved to {output_dir}")
    cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
