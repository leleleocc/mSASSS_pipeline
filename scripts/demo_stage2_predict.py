#!/usr/bin/env python3
"""Demo: load Stage-2 model and visualize scoring predictions on a few samples."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stage2.data import (
    ZhongriVUDataset,
    assign_patient_folds,
    decode_image,
    extract_vu_crop,
    load_zhongri_samples,
    select_samples,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from src.stage2.model import VUOrdinalEfficientNet, decode_scores


def load_model(weights_path: Path, device: torch.device) -> VUOrdinalEfficientNet:
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    args_dict = checkpoint.get("args", {})
    use_roi = bool(args_dict.get("use_roi", False))
    model = VUOrdinalEfficientNet(
        pretrained=False,
        hidden_dim=int(args_dict.get("hidden_dim", 256)),
        dropout=float(args_dict.get("dropout", 0.4)),
        use_roi=use_roi,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    model.to(device)
    return model


def predict_all_val(mode: str = "all") -> None:
    """Predict all validation samples across 5 folds and write results to CSV."""
    import csv as csv_mod

    source = Path("/paddle/lv/mSASSS_pipeline/data/raw_data/zhongri/1-after-trim")
    model_dir = Path("/paddle/lv/mSASSS_pipeline/src/stage2/outputs/efficientnet")
    output_path = Path("/paddle/lv/mSASSS_pipeline/scripts/stage2_val_predictions.csv")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    samples = load_zhongri_samples(source)
    assignments = assign_patient_folds(samples, 5, 42)

    rows = []
    for fold in range(5):
        weights_path = model_dir / f"fold_{fold}" / "weights" / "best.pt"
        if not weights_path.exists():
            print(f"[WARN] fold {fold} weights not found, skipping")
            continue

        model = load_model(weights_path, device)
        val_samples = [s for s in samples if assignments[s.patient_id] == fold]
        dataset = ZhongriVUDataset(val_samples, augment=False, crop_size=256)

        print(f"Fold {fold}: predicting {len(val_samples)} VUs...")
        for idx in range(len(val_samples)):
            sample = val_samples[idx]
            batch = dataset[idx]
            image_tensor = batch["image"].unsqueeze(0).to(device)
            point_xy = batch["point_xy"].unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(image_tensor, point_xy)
                scores = outputs["scores"][0].cpu().numpy()
                probs = outputs["probabilities"][0].cpu().numpy()

            rows.append({
                "sample_id": sample.sample_id,
                "patient_id": sample.patient_id,
                "view": sample.view,
                "level_name": sample.level_name,
                "level_index": sample.level_index,
                "fold": fold,
                "gt_up": sample.up_score,
                "gt_down": sample.down_score,
                "pred_up": int(scores[0]),
                "pred_down": int(scores[1]),
                "prob_up_ge1": f"{probs[0][0]:.4f}",
                "prob_up_ge2": f"{probs[0][1]:.4f}",
                "prob_up_ge3": f"{probs[0][2]:.4f}",
                "prob_down_ge1": f"{probs[1][0]:.4f}",
                "prob_down_ge2": f"{probs[1][1]:.4f}",
                "prob_down_ge3": f"{probs[1][2]:.4f}",
            })

    # Write CSV
    fields = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv_mod.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    gt_up = np.array([r["gt_up"] for r in rows])
    gt_dn = np.array([r["gt_down"] for r in rows])
    pd_up = np.array([r["pred_up"] for r in rows])
    pd_dn = np.array([r["pred_down"] for r in rows])

    print(f"\n{'='*50}")
    print(f"Total samples: {len(rows)}")
    print(f"Results written to: {output_path}")
    print(f"\nOverall metrics (5-fold cross-val):")
    print(f"  Up MAE:   {np.abs(gt_up - pd_up).mean():.4f}")
    print(f"  Down MAE: {np.abs(gt_dn - pd_dn).mean():.4f}")
    print(f"  Mean MAE: {(np.abs(gt_up - pd_up).mean() + np.abs(gt_dn - pd_dn).mean()) / 2:.4f}")
    print(f"  Up exact:   {(gt_up == pd_up).mean():.4f}")
    print(f"  Down exact: {(gt_dn == pd_dn).mean():.4f}")
    print(f"  Mean exact: {((gt_up == pd_up).sum() + (gt_dn == pd_dn).sum()) / (2*len(rows)):.4f}")
    print(f"{'='*50}")


def main() -> None:
    predict_all_val()


if __name__ == "__main__":
    main()
