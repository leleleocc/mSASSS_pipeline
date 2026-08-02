#!/usr/bin/env python3
"""Index scored Zhongri VUs and visualize online Stage-2 augmentation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
STAGE2_ROOT = Path(__file__).resolve().parent
MPL_CONFIG = STAGE2_ROOT / ".matplotlib"
MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.stage2.data import (  # noqa: E402
    AugmentationConfig,
    CropJitter,
    VUSample,
    assign_patient_folds,
    augment_intensity,
    decode_image,
    encode_image,
    extract_vu_crop,
    load_zhongri_samples,
    sample_crop_jitter,
)


DEFAULT_SOURCE = PROJECT_ROOT / "raw_data/zhongri/1-after-trim"
DEFAULT_OUTPUT = STAGE2_ROOT / "data/zhongri_vu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--preview-variants", type=int, default=3)
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
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def relative_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def write_manifest(
    path: Path,
    samples: Sequence[VUSample],
    folds: dict[str, int],
) -> None:
    fields = [
        "sample_id",
        "patient_id",
        "view",
        "level_index",
        "level_name",
        "image_id",
        "annotation_id",
        "image_path",
        "up_score",
        "down_score",
        "fold",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "sample_id": sample.sample_id,
                    "patient_id": sample.patient_id,
                    "view": sample.view,
                    "level_index": sample.level_index,
                    "level_name": sample.level_name,
                    "image_id": sample.image_id,
                    "annotation_id": sample.annotation_id,
                    "image_path": relative_to_project(sample.image_path),
                    "up_score": sample.up_score,
                    "down_score": sample.down_score,
                    "fold": folds[sample.patient_id],
                }
            )


def counter_dict(values: Sequence[int]) -> dict[str, int]:
    counts = Counter(values)
    return {str(score): int(counts.get(score, 0)) for score in range(4)}


def summarize(
    samples: Sequence[VUSample],
    folds: dict[str, int],
    augmentation: AugmentationConfig,
    crop_size: int,
    source: Path,
) -> dict[str, Any]:
    by_view: dict[str, list[VUSample]] = defaultdict(list)
    by_level: dict[str, list[VUSample]] = defaultdict(list)
    by_fold: dict[int, list[VUSample]] = defaultdict(list)
    fold_patients: dict[int, set[str]] = defaultdict(set)
    for sample in samples:
        by_view[sample.view].append(sample)
        by_level[f"{sample.view}:{sample.level_name}"].append(sample)
        fold = folds[sample.patient_id]
        by_fold[fold].append(sample)
        fold_patients[fold].add(sample.patient_id)

    def score_summary(group: Sequence[VUSample]) -> dict[str, Any]:
        return {
            "vus": len(group),
            "corners": len(group) * 2,
            "up": counter_dict([sample.up_score for sample in group]),
            "down": counter_dict([sample.down_score for sample in group]),
        }

    return {
        "source": relative_to_project(source),
        "patients": len({sample.patient_id for sample in samples}),
        "images": len({sample.image_path for sample in samples}),
        "vus": len(samples),
        "corner_labels": len(samples) * 2,
        "crop_size": crop_size,
        "orientation": {
            "left": "anterior",
            "right": "posterior",
            "top": "superior",
            "bottom": "inferior",
        },
        "labels": {
            "up": "superior spatial corner score, integer 0..3",
            "down": "inferior spatial corner score, integer 0..3",
        },
        "augmentation": {
            **asdict(augmentation),
            "online_only": True,
            "train_only": True,
            "forbidden": [
                "horizontal_flip",
                "vertical_flip",
                "mixup",
                "cutmix",
                "random_erasing",
                "elastic_deformation",
            ],
        },
        "overall": score_summary(samples),
        "by_view": {
            key: score_summary(value) for key, value in sorted(by_view.items())
        },
        "by_level": {
            key: score_summary(value) for key, value in sorted(by_level.items())
        },
        "folds": {
            str(fold): {
                "patients": len(fold_patients[fold]),
                **score_summary(by_fold[fold]),
            }
            for fold in sorted(by_fold)
        },
    }


def choose_preview_samples(samples: Sequence[VUSample]) -> list[VUSample]:
    selected: list[VUSample] = []
    used: set[str] = set()
    for view in ("C", "L"):
        candidates = [sample for sample in samples if sample.view == view]
        for target_grade in range(4):
            ranked = sorted(
                candidates,
                key=lambda sample: (
                    abs(max(sample.up_score, sample.down_score) - target_grade),
                    sample.sample_id in used,
                    sample.patient_id,
                    sample.level_index,
                ),
            )
            sample = next(value for value in ranked if value.sample_id not in used)
            selected.append(sample)
            used.add(sample.sample_id)
    return selected


def labeled_tile(image: np.ndarray, title: str, tile_size: int) -> np.ndarray:
    image = cv2.resize(image, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
    header = 34
    tile = cv2.copyMakeBorder(
        image,
        header,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(20, 20, 20),
    )
    cv2.putText(
        tile,
        title,
        (7, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return tile


def write_preview(
    path: Path,
    samples: Sequence[VUSample],
    augmentation: AugmentationConfig,
    crop_size: int,
    variants: int,
    seed: int,
) -> None:
    if variants < 1:
        raise ValueError("preview variants must be positive")
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    for sample in choose_preview_samples(samples):
        image = decode_image(sample.image_path)
        canonical, _ = extract_vu_crop(image, sample, crop_size, CropJitter())
        tiles = [
            labeled_tile(
                canonical,
                f"{sample.sample_id} up={sample.up_score} down={sample.down_score}",
                crop_size,
            )
        ]
        for index in range(variants):
            jitter = sample_crop_jitter(rng, augmentation)
            augmented, _ = extract_vu_crop(image, sample, crop_size, jitter)
            augmented = augment_intensity(augmented, rng, augmentation)
            tiles.append(labeled_tile(augmented, f"aug {index + 1}", crop_size))
        rows.append(np.concatenate(tiles, axis=1))
    contact_sheet = np.concatenate(rows, axis=0)
    encode_image(path, contact_sheet, quality=94)


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.exist_ok:
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
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
    samples = load_zhongri_samples(args.source)
    folds = assign_patient_folds(samples, n_folds=args.folds, seed=args.seed)
    write_manifest(output / "manifest.csv", samples, folds)
    summary = summarize(samples, folds, augmentation, args.crop_size, args.source)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_preview(
        output / "augmentation_preview.jpg",
        samples,
        augmentation,
        args.crop_size,
        args.preview_variants,
        args.seed,
    )
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    print(f"Stage-2 data index saved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
