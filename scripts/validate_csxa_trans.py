#!/usr/bin/env python3
"""Validate the transformed CSXA LabelMe dataset and estimated T1 geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from migrate_csxa_with_t1 import SOURCE_LABELS, T1_LABELS
from visualize_csxa_keypoints import load_points
from visualize_csxa_t1_candidates import estimate_t1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("raw_data/CSXA/CSXA-trans"),
    )
    parser.add_argument(
        "--source-image-dir",
        type=Path,
        default=Path("raw_data/CSXA/origin/datasets-PNG"),
    )
    return parser.parse_args()


def normalized_label(shape: dict[str, Any]) -> str:
    return " ".join(str(shape.get("label", "")).strip().casefold().split())


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    json_dir = root / "datasets-JSON"
    image_dir = root / "datasets-PNG"
    source_image_dir = args.source_image_dir.resolve()
    json_paths = sorted(json_dir.glob("*.json"))
    expected_labels = SOURCE_LABELS | {label.casefold() for label in T1_LABELS.values()}
    errors: list[str] = []
    outside_cases: set[str] = set()
    outside_points = 0
    hardlinked_images = 0
    maximum_geometry_error = 0.0

    for json_path in json_paths:
        try:
            annotation = json.loads(json_path.read_text(encoding="utf-8"))
            shapes = annotation.get("shapes", [])
            labels = [normalized_label(shape) for shape in shapes]
            if len(shapes) != 26 or len(set(labels)) != 26:
                raise RuntimeError(f"expected 26 unique shapes, got {len(shapes)}")
            if set(labels) != expected_labels:
                missing = sorted(expected_labels - set(labels))
                extra = sorted(set(labels) - expected_labels)
                raise RuntimeError(f"label mismatch: missing={missing}, extra={extra}")
            if any(
                shape.get("shape_type") != "point"
                or len(shape.get("points", [])) != 1
                or len(shape["points"][0]) != 2
                for shape in shapes
            ):
                raise RuntimeError("not all shapes are canonical single points")
            if annotation.get("imageData", "missing") is not None:
                raise RuntimeError("imageData is not null")

            image_name = f"{json_path.stem}.png"
            if annotation.get("imagePath") != image_name:
                raise RuntimeError(f"incorrect imagePath: {annotation.get('imagePath')}")
            image_path = image_dir / image_name
            source_image_path = source_image_dir / image_name
            if not image_path.is_file():
                raise RuntimeError("paired output image is missing")
            if source_image_path.is_file() and image_path.samefile(source_image_path):
                hardlinked_images += 1

            records = load_points(annotation)
            estimated = estimate_t1(records)
            lookup = {normalized_label(shape): shape for shape in shapes}
            width = int(annotation["imageWidth"])
            height = int(annotation["imageHeight"])
            for key, label in T1_LABELS.items():
                shape = lookup[label.casefold()]
                actual = np.asarray(shape["points"][0], dtype=np.float64)
                error = float(np.linalg.norm(actual - estimated[key]))
                maximum_geometry_error = max(maximum_geometry_error, error)
                outside = not (0.0 <= actual[0] < width and 0.0 <= actual[1] < height)
                flag = bool(shape.get("flags", {}).get("outside_image", False))
                if flag != outside:
                    raise RuntimeError(f"incorrect outside_image flag for {label}")
                if outside:
                    outside_points += 1
                    outside_cases.add(json_path.stem)
        except Exception as error:
            errors.append(f"{json_path.stem}: {error}")

    image_count = sum(1 for _ in image_dir.glob("*.png"))
    summary = {
        "json_files": len(json_paths),
        "image_files": image_count,
        "hardlinked_images": hardlinked_images,
        "outside_image_cases": len(outside_cases),
        "outside_image_points": outside_points,
        "maximum_t1_geometry_error_px": maximum_geometry_error,
        "errors": len(errors),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for error in errors[:30]:
        print(f"ERROR {error}")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
