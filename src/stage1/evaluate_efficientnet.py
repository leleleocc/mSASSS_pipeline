#!/usr/bin/env python3
"""Evaluate Stage-1 EfficientNet landmarks in original-image coordinates."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MPL_CONFIG = Path(__file__).resolve().parent / ".matplotlib"
MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.stage1.evaluation_common import CasePrediction, save_evaluation  # noqa: E402
from src.stage1.efficientnet import (  # noqa: E402
    SpineKeypointDataset,
    EfficientNetKeypointModel,
    decode_original_coordinates,
    normalize_image_size,
    seed_worker,
)


STAGE1_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = STAGE1_ROOT / "data/spine_keypoints_19pt/data.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, nargs="+", default=None, metavar="N")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--project",
        type=Path,
        default=STAGE1_ROOT / "outputs/evaluation/efficientnet",
    )
    parser.add_argument("--name", default="efficientnet_val")
    parser.add_argument("--save-all", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-worst", type=int, default=20)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exist-ok", action="store_true")
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


def main() -> int:
    args = parse_args()
    weights = args.weights.resolve()
    data_yaml = args.data.resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {weights}")
    if not data_yaml.is_file():
        raise FileNotFoundError(f"dataset YAML does not exist: {data_yaml}")
    if args.batch < 1 or args.workers < 0 or args.save_worst < 0:
        raise ValueError("batch must be positive; workers/save-worst cannot be negative")
    output_dir = (args.project / args.name).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.exist_ok:
        raise FileExistsError(f"evaluation directory is not empty: {output_dir}")

    device = resolve_device(args.device)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise RuntimeError("expected an EfficientNet Stage-1 checkpoint with model_state")
    checkpoint_args = checkpoint.get("args", {})
    image_size = normalize_image_size(
        args.imgsz or checkpoint_args.get("imgsz", [1024, 768])
    )
    if any(value < 64 or value % 32 for value in image_size):
        raise ValueError("each input dimension must be >=64 and divisible by 32")

    dataset = SpineKeypointDataset(data_yaml, args.split, image_size, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        persistent_workers=args.workers > 0,
    )
    names = dataset.config.get("names", {})
    model = EfficientNetKeypointModel(
        num_classes=len(names) if names else 2,
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    amp_enabled = args.amp and device.type == "cuda"
    cases: list[CasePrediction] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model(images)
            predicted_points = decode_original_coordinates(
                outputs["coordinates"].cpu(),
                image_size,
                batch["resize_scale"],
                batch["pad"],
                batch["original_shape"],
            ).numpy()
            probabilities = outputs["class_logits"].softmax(dim=1).cpu()
            predicted_classes = probabilities.argmax(dim=1)
            confidences = probabilities.max(dim=1).values
            for index, path in enumerate(batch["path"]):
                cases.append(
                    CasePrediction(
                        image_path=Path(path),
                        gt_class=int(batch["class_id"][index]),
                        gt_points=batch["original_points"][index].numpy().astype("float64"),
                        pred_class=int(predicted_classes[index]),
                        pred_points=predicted_points[index],
                        confidence=float(confidences[index]),
                    )
                )

    summary = save_evaluation(
        cases,
        names,
        dataset.label_dir.parent.parent,
        output_dir,
        model_metadata={
            "type": "EfficientNet-B0-FPN",
            "weights": str(weights),
            "checkpoint_epoch": (
                int(checkpoint["epoch"]) + 1 if "epoch" in checkpoint else None
            ),
            "checkpoint_metrics": checkpoint.get("metrics"),
            "split": args.split,
            "imgsz_hw": list(image_size),
            "amp": amp_enabled,
        },
        save_all=args.save_all,
        save_worst=args.save_worst,
    )
    print(json.dumps(summary["point_metrics"], ensure_ascii=False, indent=2))
    print(f"Results saved to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
