#!/usr/bin/env python3
"""Create a CSXA copy with estimated T1 landmarks and no embedded image data."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from visualize_csxa_keypoints import load_points
from visualize_csxa_t1_candidates import estimate_t1


T1_LABELS = {
    "T1-TL?": "T1 top left",
    "T1-TR?": "T1 top right",
    "T1-CTR?": "T1 centroid",
}
ESTIMATE_DESCRIPTION = "estimated from C6/C7 geometry"
SOURCE_LABELS = {
    "c2 centroid",
    "c2 bottom left",
    "c2 bottom right",
    *{
        f"c{level} {position}"
        for level in range(3, 8)
        for position in ("top left", "top right", "bottom left", "bottom right")
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=Path("raw_data/CSXA/origin/datasets-JSON"),
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("raw_data/CSXA/origin/datasets-PNG"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("raw_data/CSXA/CSXA-trans"),
    )
    parser.add_argument(
        "--image-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="How to place source images in the transformed dataset.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all cases and report statistics without writing files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing transformed JSON files.",
    )
    return parser.parse_args()


def normalized_label(shape: dict[str, Any]) -> str:
    return " ".join(str(shape.get("label", "")).strip().casefold().split())


def select_duplicate_landmark(
    label: str,
    candidates: list[np.ndarray],
) -> np.ndarray:
    """Resolve duplicate landmark labels using their explicit left/right meaning."""
    if len(candidates) == 1:
        return candidates[0]
    if label.endswith(" left"):
        return min(candidates, key=lambda point: float(point[0]))
    if label.endswith(" right"):
        return max(candidates, key=lambda point: float(point[0]))
    return np.median(np.asarray(candidates), axis=0)


def point_shape(
    label: str,
    point: np.ndarray,
    width: int,
    height: int,
) -> dict[str, Any]:
    outside_image = not (
        0.0 <= float(point[0]) < width and 0.0 <= float(point[1]) < height
    )
    return {
        "description": ESTIMATE_DESCRIPTION,
        "label": label,
        "points": [[float(point[0]), float(point[1])]],
        "group_id": None,
        "shape_type": "point",
        "flags": {"estimated": True, "outside_image": outside_image},
    }


def transform_annotation(
    annotation: dict[str, Any],
    image_name: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    records = load_points(annotation)
    candidates = estimate_t1(records)
    output = dict(annotation)
    width = int(annotation["imageWidth"])
    height = int(annotation["imageHeight"])
    t1_names = {label.casefold() for label in T1_LABELS.values()}
    record_groups: dict[str, list[np.ndarray]] = {}
    for record in records:
        label = " ".join(record["label"].strip().casefold().split())
        if label in SOURCE_LABELS:
            record_groups.setdefault(label, []).append(record["xy"])
    missing_records = sorted(SOURCE_LABELS - record_groups.keys())
    if missing_records:
        raise RuntimeError(f"missing readable source landmarks: {missing_records}")
    selected_points = {
        label: select_duplicate_landmark(label, points)
        for label, points in record_groups.items()
    }

    # Canonicalize every source landmark as one LabelMe point. This repairs the
    # known near-duplicate polygon vertices and removes duplicate label clicks.
    cleaned_shapes: list[dict[str, Any]] = []
    emitted: set[str] = set()
    for source_shape in annotation.get("shapes", []):
        label = normalized_label(source_shape)
        if label in t1_names or label in emitted:
            continue
        shape = dict(source_shape)
        if label in SOURCE_LABELS:
            point = selected_points[label]
            shape["points"] = [[float(point[0]), float(point[1])]]
            shape["shape_type"] = "point"
            emitted.add(label)
        cleaned_shapes.append(shape)
    output["shapes"] = cleaned_shapes
    output["shapes"].extend(
        point_shape(T1_LABELS[key], candidates[key], width, height) for key in T1_LABELS
    )
    output["imagePath"] = image_name
    output["imageData"] = None
    return output, candidates


def place_image(source: Path, destination: Path, mode: str) -> str:
    if destination.exists():
        if source.samefile(destination):
            return "existing_hardlink"
        if source.stat().st_size == destination.stat().st_size:
            return "existing_file"
        raise RuntimeError(f"destination image differs from source: {destination}")
    if mode == "copy":
        shutil.copy2(source, destination)
        return "copied"
    try:
        os.link(source, destination)
        return "hardlinked"
    except OSError:
        shutil.copy2(source, destination)
        return "copied_fallback"


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def point_is_outside(point: np.ndarray, width: int, height: int) -> bool:
    x, y = (float(point[0]), float(point[1]))
    return not (0.0 <= x < width and 0.0 <= y < height)


def main() -> int:
    args = parse_args()
    json_dir = args.json_dir.resolve()
    image_dir = args.image_dir.resolve()
    output_root = args.output_root.resolve()
    json_paths = sorted(json_dir.glob("*.json"))
    if not json_paths:
        raise RuntimeError(f"no JSON files found in {json_dir}")

    output_json_dir = output_root / "datasets-JSON"
    output_image_dir = output_root / "datasets-PNG"
    if not args.dry_run:
        output_json_dir.mkdir(parents=True, exist_ok=True)
        output_image_dir.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    outside_cases: list[dict[str, Any]] = []
    label_anomaly_cases: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for index, json_path in enumerate(json_paths, start=1):
        try:
            image_name = f"{json_path.stem}.png"
            image_path = image_dir / image_name
            if not image_path.is_file():
                raise RuntimeError(f"paired image does not exist: {image_path}")
            annotation = json.loads(json_path.read_text(encoding="utf-8"))
            source_label_counts = Counter(
                normalized_label(shape) for shape in annotation.get("shapes", [])
            )
            missing_labels = sorted(SOURCE_LABELS - source_label_counts.keys())
            unexpected_labels = sorted(source_label_counts.keys() - SOURCE_LABELS)
            duplicate_labels = sorted(
                label for label, count in source_label_counts.items() if count != 1
            )
            if missing_labels or unexpected_labels or duplicate_labels:
                label_anomaly_cases.append(
                    {
                        "case": json_path.stem,
                        "missing": missing_labels,
                        "unexpected": unexpected_labels,
                        "duplicates": duplicate_labels,
                    }
                )
            transformed, candidates = transform_annotation(annotation, image_name)
            width = int(annotation["imageWidth"])
            height = int(annotation["imageHeight"])
            outside = [
                {
                    "label": T1_LABELS[key],
                    "point": [float(point[0]), float(point[1])],
                }
                for key, point in candidates.items()
                if point_is_outside(point, width, height)
            ]
            if outside:
                outside_cases.append(
                    {
                        "case": json_path.stem,
                        "image_size": [width, height],
                        "points": outside,
                    }
                )

            counts["validated"] += 1
            counts["t1_points"] += len(candidates)
            counts["base64_removed"] += int(transformed.get("imageData") is None)
            counts["output_shapes"] += len(transformed["shapes"])
            counts[f"cases_with_{len(transformed['shapes'])}_shapes"] += 1

            if not args.dry_run:
                output_json_path = output_json_dir / json_path.name
                if output_json_path.exists() and not args.overwrite:
                    raise RuntimeError(
                        f"output JSON already exists; use --overwrite: {output_json_path}"
                    )
                image_status = place_image(
                    image_path, output_image_dir / image_name, args.image_mode
                )
                counts[image_status] += 1
                write_json_atomic(output_json_path, transformed)
                counts["json_written"] += 1
        except Exception as error:  # continue to produce a complete audit
            errors.append({"case": json_path.stem, "error": str(error)})

        if index % 500 == 0 or index == len(json_paths):
            print(f"processed {index}/{len(json_paths)}", flush=True)

    report = {
        "source_json_dir": str(json_dir),
        "source_image_dir": str(image_dir),
        "output_root": str(output_root),
        "dry_run": args.dry_run,
        "image_mode": args.image_mode,
        "formula": {
            "T1 top left": "extend C7 TL->BL by distance(C6 BL, C7 TL)",
            "T1 top right": "extend C7 TR->BR by distance(C6 BR, C7 TR)",
            "T1 centroid": (
                "transfer C7 centroid projection fraction and signed perpendicular "
                "distance from the C7 top edge to the estimated T1 top edge"
            ),
        },
        "counts": dict(sorted(counts.items())),
        "outside_image_cases": outside_cases,
        "label_anomaly_cases": label_anomaly_cases,
        "errors": errors,
    }
    if not args.dry_run:
        write_json_atomic(output_root / "migration_report.json", report)
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))
    print(f"outside-image cases: {len(outside_cases)}")
    for item in outside_cases[:20]:
        print(f"OUTSIDE {json.dumps(item, ensure_ascii=False)}")
    print(f"label-anomaly cases: {len(label_anomaly_cases)}")
    for item in label_anomaly_cases[:20]:
        print(f"LABEL_ANOMALY {json.dumps(item, ensure_ascii=False)}")
    print(f"errors: {len(errors)}")
    if errors:
        for item in errors[:20]:
            print(f"ERROR {item['case']}: {item['error']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
