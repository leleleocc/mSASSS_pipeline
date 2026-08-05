#!/usr/bin/env python3
"""Run full Stage-1 → Stage-2 pipeline on a BUU image and visualize mSASSS predictions."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import torch

from src.stage1.efficientnet import (
    EfficientNetKeypointModel,
    IMAGENET_MEAN as S1_MEAN,
    IMAGENET_STD as S1_STD,
    letterbox,
    normalize_image_size,
)
from src.stage2.data import (
    DEFAULT_CROP_SIZE,
    IMAGENET_MEAN as S2_MEAN,
    IMAGENET_STD as S2_STD,
    LUMBAR_LEVELS,
    VUSample,
    crop_quad,
    extract_vu_crop,
)
from src.stage2.model import VUOrdinalEfficientNet

# --- Paths ---
STAGE1_WEIGHTS = Path("src/stage1/outputs/efficientnet/efficientnet_b0_19pt_bs32/weights/best.pt")
STAGE2_WEIGHTS = Path("src/stage2/outputs/efficientnet/fold_0_noROI/weights/best.pt")
IMAGE_PATH = Path("src/stage1/data/spine_keypoints_19pt/images/val/buu__0001-F-037Y1.jpg")
OUTPUT_PATH = Path("src/stage1/outputs/vis_compare/buu__0001_msasss_demo.jpg")
IMGSZ = [1024, 768]

# Keypoint indices: B1-B7 = indices 0-6; V1P1,V1P2,...V6P1,V6P2 = indices 7-18
# VU i (0-based) uses upper_center=B[i+1], lower_center=B[i+2], up_corner=V(i+1)P1, down_corner=V(i+1)P2
# Mapping: VU[0]=T12-L1 uses B2,B3, V1P1,V1P2  (B indices 1,2; corner indices 7,8)
# Wait - let me reconsider. The 19pt Stage-1 keypoints are:
# B1..B7 (7 centroids for T12 through S1), V1P1,V1P2..V6P1,V6P2 (12 corners for 6 VUs)
# VU[i] (i=0..5): upper_center=B[i+1], lower_center=B[i+2], up_corner=V(i+1)P1, down_corner=V(i+1)P2
# Indices: B1=0..B7=6, V1P1=7,V1P2=8, V2P1=9,V2P2=10, ... V6P1=17,V6P2=18

def build_vu_sample(kpts: np.ndarray, vu_index: int, image_path: Path, w: int, h: int) -> VUSample:
    """Build a VUSample from Stage-1 predicted keypoints for a given VU index (0-5)."""
    # B1..B7 = kpts[0..6], VU corners: V(i+1)P1 = kpts[7 + 2*i], V(i+1)P2 = kpts[8 + 2*i]
    upper_center = tuple(kpts[vu_index + 1].tolist())  # B(i+2) for disc between vertebra i+1 and i+2
    lower_center = tuple(kpts[vu_index + 2].tolist())
    up_corner = tuple(kpts[7 + 2 * vu_index].tolist())
    down_corner = tuple(kpts[8 + 2 * vu_index].tolist())
    return VUSample(
        sample_id=f"demo_vu{vu_index}",
        patient_id="buu_0001",
        view="L",
        level_index=vu_index + 1,
        level_name=LUMBAR_LEVELS[vu_index],
        image_id=0,
        annotation_id=0,
        image_path=image_path,
        image_width=w,
        image_height=h,
        upper_center=upper_center,
        lower_center=lower_center,
        up_corner=up_corner,
        down_corner=down_corner,
        up_score=0,
        down_score=0,
    )


def predict_stage2(model: VUOrdinalEfficientNet, crop_bgr: np.ndarray, device: torch.device) -> tuple[int, int]:
    """Run Stage-2 inference on a 256x256 VU crop, return (up_score, down_score)."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - S2_MEAN) / S2_STD
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda"):
        outputs = model(tensor)
    scores = outputs["scores"][0].cpu().numpy()  # [up, down]
    return int(scores[0]), int(scores[1])


def predict_stage1(model: EfficientNetKeypointModel, image_bgr: np.ndarray, device: torch.device) -> np.ndarray:
    """Run Stage-1 EfficientNet inference, return 19x2 pixel coords."""
    h_orig, w_orig = image_bgr.shape[:2]
    dummy_pts = np.zeros((19, 2), dtype=np.float32)
    canvas, _, scale, (pad_x, pad_y) = letterbox(image_bgr, dummy_pts, IMGSZ)
    image_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image_rgb = (image_rgb - S1_MEAN) / S1_STD
    tensor = torch.from_numpy(image_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda"):
        outputs = model(tensor)
    coords = outputs["coordinates"][0].cpu().numpy()  # (19, 2) normalized
    input_h, input_w = normalize_image_size(IMGSZ)
    coords[:, 0] *= input_w
    coords[:, 1] *= input_h
    coords[:, 0] = (coords[:, 0] - pad_x) / scale
    coords[:, 1] = (coords[:, 1] - pad_y) / scale
    return coords


SCORE_COLORS = {
    0: (0, 200, 0),     # green = normal
    1: (0, 200, 255),   # yellow-orange
    2: (0, 100, 255),   # orange
    3: (0, 0, 255),     # red = severe
}


def draw_results(image: np.ndarray, kpts: np.ndarray, vu_scores: list[tuple[str, int, int]]) -> np.ndarray:
    """Draw keypoints and mSASSS scores on the image."""
    vis = image.copy()
    # Draw centroids B1-B7
    for i in range(7):
        x, y = int(kpts[i, 0]), int(kpts[i, 1])
        cv2.circle(vis, (x, y), 5, (255, 255, 0), -1)
        cv2.putText(vis, f"B{i+1}", (x + 8, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    # Draw VU corners
    for i in range(6):
        for j in range(2):
            idx = 7 + 2 * i + j
            x, y = int(kpts[idx, 0]), int(kpts[idx, 1])
            cv2.circle(vis, (x, y), 4, (255, 0, 255), -1)
    # Draw mSASSS scores near each VU
    for vu_idx, (level_name, up_s, down_s) in enumerate(vu_scores):
        # Position label between the two corners of this VU
        cx = int((kpts[7 + 2 * vu_idx, 0] + kpts[8 + 2 * vu_idx, 0]) / 2)
        cy = int((kpts[7 + 2 * vu_idx, 1] + kpts[8 + 2 * vu_idx, 1]) / 2)
        label = f"{level_name}: up={up_s} dn={down_s}"
        color = SCORE_COLORS.get(max(up_s, down_s), (255, 255, 255))
        cv2.putText(vis, label, (cx + 15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    # Legend
    h = vis.shape[0]
    cv2.putText(vis, "mSASSS: 0=normal 1=sclerosis 2=erosion 3=syndesmophyte", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return vis


def main() -> int:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load Stage-1
    s1_model = EfficientNetKeypointModel(pretrained=False)
    ckpt1 = torch.load(STAGE1_WEIGHTS, map_location="cpu", weights_only=False)
    s1_model.load_state_dict(ckpt1["model_state"])
    s1_model.to(device).eval()
    print(f"Stage-1 loaded: {STAGE1_WEIGHTS} (epoch {ckpt1['epoch']+1})")

    # Load Stage-2
    s2_model = VUOrdinalEfficientNet(pretrained=False, use_roi=False)
    ckpt2 = torch.load(STAGE2_WEIGHTS, map_location="cpu", weights_only=False)
    s2_model.load_state_dict(ckpt2["model_state"])
    s2_model.to(device).eval()
    print(f"Stage-2 loaded: {STAGE2_WEIGHTS} (epoch {ckpt2['epoch']+1})")

    # Load image
    image_bgr = cv2.imdecode(np.fromfile(IMAGE_PATH, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Cannot read image: {IMAGE_PATH}")
    h, w = image_bgr.shape[:2]
    print(f"Image: {IMAGE_PATH.name} ({w}x{h})")

    # Stage-1: predict 19 keypoints
    kpts = predict_stage1(s1_model, image_bgr, device)
    print("Stage-1 keypoints predicted:")
    names = [f"B{i+1}" for i in range(7)] + [f"V{i//2+1}P{i%2+1}" for i in range(12)]
    for i, name in enumerate(names):
        print(f"  {name}: ({kpts[i,0]:.1f}, {kpts[i,1]:.1f})")

    # Stage-2: predict mSASSS for each VU
    vu_scores = []
    for vu_idx in range(6):
        sample = build_vu_sample(kpts, vu_idx, IMAGE_PATH, w, h)
        crop, _ = extract_vu_crop(image_bgr, sample, DEFAULT_CROP_SIZE, jitter=None)
        up_s, down_s = predict_stage2(s2_model, crop, device)
        level = LUMBAR_LEVELS[vu_idx]
        vu_scores.append((level, up_s, down_s))
        print(f"  {level}: up={up_s}, down={down_s}")

    # Visualize
    vis = draw_results(image_bgr, kpts, vu_scores)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".jpg", vis)[1].tofile(OUTPUT_PATH)
    print(f"\nSaved: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
