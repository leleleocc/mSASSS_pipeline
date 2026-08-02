#!/usr/bin/env python3
"""Evaluate Stage-1 YOLO-Pose landmarks in original-image coordinates."""

from __future__ import annotations

import argparse
import json
import os
import sys
from functools import partial
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MPL_CONFIG = Path(__file__).resolve().parent / ".matplotlib"
MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG))

import numpy as np  # noqa: E402

from src.stage1.evaluation_common import (  # noqa: E402
    CasePrediction,
    json_safe,
    read_pose_label,
    resolve_split,
    save_evaluation,
)


STAGE1_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = STAGE1_ROOT / "data/spine_keypoints_19pt/data.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="fixed input height width; defaults to checkpoint metadata or 1024 768",
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument(
        "--project",
        type=Path,
        default=STAGE1_ROOT / "outputs/evaluation/yolo",
    )
    parser.add_argument("--name", default="yolo_val")
    parser.add_argument("--save-all", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-worst", type=int, default=20)
    parser.add_argument(
        "--official-val",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also run Ultralytics box/pose mAP validation",
    )
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


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
    if not 0 <= args.conf <= 1:
        raise ValueError("conf must be in [0, 1]")
    output_dir = (args.project / args.name).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.exist_ok:
        raise FileExistsError(f"evaluation directory is not empty: {output_dir}")

    config, image_paths, label_dir, dataset_root = resolve_split(data_yaml, args.split)
    names = config.get("names", {})
    # Import custom classes before torch loads a Stage-1 checkpoint.
    from src.stage1.yolo_strategy import (
        NormalizedMREPoseValidator,
        normalize_input_shape,
    )
    from ultralytics import YOLO

    model = YOLO(str(weights))
    checkpoint_shape = getattr(model.model, "stage1_input_shape", (1024, 768))
    input_shape = normalize_input_shape(args.imgsz or checkpoint_shape)
    official_metrics: dict[str, object] | None = None
    if args.official_val:
        official = model.val(
            validator=partial(
                NormalizedMREPoseValidator,
                input_shape=input_shape,
            ),
            data=str(data_yaml),
            split=args.split,
            imgsz=max(input_shape),
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            rect=False,
            plots=True,
            project=str(output_dir),
            name="official_yolo",
            exist_ok=True,
        )
        official_metrics = json_safe(official.results_dict)

    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=list(input_shape),
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        max_det=1,
        rect=False,
        stream=True,
        verbose=False,
    )
    cases: list[CasePrediction] = []
    for image_path, result in zip(image_paths, results, strict=True):
        height, width = result.orig_shape
        gt_class, gt_points = read_pose_label(
            label_dir / f"{image_path.stem}.txt",
            width,
            height,
        )
        predicted_points: np.ndarray | None = None
        predicted_class: int | None = None
        confidence: float | None = None
        if result.boxes is not None and len(result.boxes):
            best = int(result.boxes.conf.argmax().item())
            predicted_class = int(result.boxes.cls[best].item())
            confidence = float(result.boxes.conf[best].item())
            if result.keypoints is None or len(result.keypoints.xy) <= best:
                raise RuntimeError(f"prediction has no keypoints: {image_path}")
            predicted_points = (
                result.keypoints.xy[best].detach().cpu().numpy().astype(np.float64)
            )
        cases.append(
            CasePrediction(
                image_path=image_path,
                gt_class=gt_class,
                gt_points=gt_points,
                pred_class=predicted_class,
                pred_points=predicted_points,
                confidence=confidence,
            )
        )

    summary = save_evaluation(
        cases,
        names,
        dataset_root,
        output_dir,
        model_metadata={
            "type": "YOLO26s-Pose",
            "weights": str(weights),
            "split": args.split,
            "imgsz_hw": list(input_shape),
            "conf": args.conf,
            "official_yolo": official_metrics,
        },
        save_all=args.save_all,
        save_worst=args.save_worst,
    )
    print(json.dumps(summary["point_metrics"], ensure_ascii=False, indent=2))
    print(f"Results saved to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
