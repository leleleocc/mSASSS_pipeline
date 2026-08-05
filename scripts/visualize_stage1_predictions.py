#!/usr/bin/env python3
"""Predict 19 keypoints with both Stage-1 models and visualize on val images."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch

from src.stage1.efficientnet import (
    EfficientNetKeypointModel,
    KEYPOINT_NAMES,
    letterbox,
    normalize_image_size,
    read_pose_label,
    resolve_dataset_split,
)


YOLO_WEIGHTS = Path("src/stage1/outputs/yolo/yolo26s_pose_19pt_bs32_lr5e4_v2/weights/best.pt")
EFFNET_WEIGHTS = Path("src/stage1/outputs/efficientnet/efficientnet_b0_19pt_bs32/weights/best.pt")
DATA_YAML = Path("src/stage1/data/spine_keypoints_19pt/data.yaml")
OUTPUT_DIR = Path("src/stage1/outputs/vis_compare")
NUM_SAMPLES = 12
IMGSZ = [1024, 768]
SEED = 42


COLORS_GT = (0, 255, 0)       # green for ground truth
COLORS_YOLO = (0, 165, 255)   # orange for YOLO
COLORS_EFFNET = (255, 0, 255) # magenta for EfficientNet


def predict_efficientnet(
    model: EfficientNetKeypointModel,
    image_bgr: np.ndarray,
    imgsz: list[int],
    device: torch.device,
) -> np.ndarray:
    """Run EfficientNet inference, return 19x2 points in original pixel coords."""
    h_orig, w_orig = image_bgr.shape[:2]
    dummy_pts = np.zeros((19, 2), dtype=np.float32)
    canvas, _, scale, (pad_x, pad_y) = letterbox(image_bgr, dummy_pts, imgsz)
    # Normalize
    image_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    from src.stage1.efficientnet import IMAGENET_MEAN, IMAGENET_STD
    image_rgb = (image_rgb - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(image_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda"):
        outputs = model(tensor)

    # outputs["coordinates"] is (1, 19, 2) in normalized [0,1] relative to input
    coords = outputs["coordinates"][0].cpu().numpy()  # (19, 2)
    input_h, input_w = normalize_image_size(imgsz)
    # Convert to input pixel coords
    coords[:, 0] *= input_w
    coords[:, 1] *= input_h
    # Undo letterbox
    coords[:, 0] = (coords[:, 0] - pad_x) / scale
    coords[:, 1] = (coords[:, 1] - pad_y) / scale
    return coords


def predict_yolo(yolo_model, image_path: Path) -> np.ndarray | None:
    """Run YOLO pose inference, return 19x2 points in original pixel coords."""
    results = yolo_model.predict(
        str(image_path), imgsz=max(IMGSZ), conf=0.25, verbose=False
    )
    if not results or len(results[0].keypoints) == 0:
        return None
    # Take best detection
    kpts = results[0].keypoints.xy[0].cpu().numpy()  # (19, 2)
    return kpts


def draw_points(image: np.ndarray, points: np.ndarray, color: tuple, radius: int = 4) -> None:
    for i, (x, y) in enumerate(points):
        if x > 0 or y > 0:
            cv2.circle(image, (int(x), int(y)), radius, color, -1)


def draw_legend(image: np.ndarray) -> None:
    h = image.shape[0]
    y_start = h - 80
    cv2.putText(image, "GT", (10, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS_GT, 2)
    cv2.putText(image, "YOLO", (10, y_start + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS_YOLO, 2)
    cv2.putText(image, "EfficientNet", (10, y_start + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS_EFFNET, 2)


def main() -> int:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load EfficientNet
    effnet = EfficientNetKeypointModel(pretrained=False)
    ckpt = torch.load(EFFNET_WEIGHTS, map_location="cpu", weights_only=False)
    effnet.load_state_dict(ckpt["model_state"])
    effnet.to(device)
    effnet.eval()
    print(f"Loaded EfficientNet from {EFFNET_WEIGHTS} (epoch {ckpt['epoch']+1})")

    # Load YOLO
    from ultralytics import YOLO
    yolo_model = YOLO(str(YOLO_WEIGHTS))
    print(f"Loaded YOLO from {YOLO_WEIGHTS}")

    # Load val split
    config, image_paths, label_dir = resolve_dataset_split(DATA_YAML, "val")

    # Sample images
    random.seed(SEED)
    selected = random.sample(image_paths, min(NUM_SAMPLES, len(image_paths)))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for img_path in selected:
        image_bgr = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"  skip (cannot decode): {img_path.name}")
            continue
        h_orig, w_orig = image_bgr.shape[:2]

        # Ground truth
        label_path = label_dir / img_path.with_suffix(".txt").name
        if not label_path.exists():
            print(f"  skip (no label): {img_path.name}")
            continue
        _, gt_points = read_pose_label(label_path, w_orig, h_orig)

        # Predictions
        yolo_points = predict_yolo(yolo_model, img_path)
        effnet_points = predict_efficientnet(effnet, image_bgr, IMGSZ, device)

        # Draw
        vis = image_bgr.copy()
        draw_points(vis, gt_points, COLORS_GT, radius=5)
        if yolo_points is not None:
            draw_points(vis, yolo_points, COLORS_YOLO, radius=4)
        draw_points(vis, effnet_points, COLORS_EFFNET, radius=3)
        draw_legend(vis)

        out_path = OUTPUT_DIR / f"{img_path.stem}_compare.jpg"
        cv2.imencode(".jpg", vis)[1].tofile(out_path)
        print(f"  saved: {out_path.name}")

    print(f"\nDone. {len(selected)} images saved to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
