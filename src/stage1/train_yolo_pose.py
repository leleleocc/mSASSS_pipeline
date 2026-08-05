#!/usr/bin/env python3
"""Train Stage-1 YOLO26s-Pose on the fixed 19-point dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.stage1.yolo_strategy import SpinePoseTrainer, normalize_input_shape


STAGE1_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = STAGE1_ROOT / "data/spine_keypoints_19pt/data.yaml"
DEFAULT_PROJECT = STAGE1_ROOT / "outputs/yolo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", default="yolo26s-pose.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, nargs="+", default=[1024, 768], metavar="N")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr0", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--structure-gain", type=float, default=1.0)
    parser.add_argument("--degrees", type=float, default=5.0)
    parser.add_argument("--translate", type=float, default=0.03)
    parser.add_argument("--scale", type=float, default=0.10)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", default="yolo26s_pose_19pt")
    parser.add_argument("--resume", nargs="?", const=True, default=False)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_shape = normalize_input_shape(args.imgsz)
    data = args.data.expanduser().resolve()
    if not data.is_file():
        raise FileNotFoundError(f"dataset YAML does not exist: {data}")
    if args.epochs < 1 or args.batch < 1 or args.workers < 0:
        raise ValueError("epochs/batch must be positive and workers cannot be negative")
    if min(args.structure_gain, args.degrees, args.translate, args.scale) < 0:
        raise ValueError("augmentation and structure-loss gains cannot be negative")

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        trainer=SpinePoseTrainer,
        data=str(data),
        epochs=args.epochs,
        imgsz=max(input_shape),
        input_shape=list(input_shape),
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        pretrained=not args.no_pretrained,
        optimizer="AdamW",
        lr0=args.lr0,
        weight_decay=args.weight_decay,
        patience=args.patience,
        spine_structure_gain=args.structure_gain,
        amp=not args.no_amp,
        val=True,
        plots=True,
        project=str(args.project.expanduser().resolve()),
        name=args.name,
        resume=args.resume,
        exist_ok=args.exist_ok,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.10,
        rle=0.0,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
