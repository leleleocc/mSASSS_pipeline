#!/usr/bin/env python3
"""Crop near-black image borders and transform a COCO keypoint dataset.

The source dataset is never modified.  The output is built in a staging
directory, validated, and atomically published only when ``--apply`` is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


DEFAULT_SOURCE = Path("raw_data/0-before-trim")
DEFAULT_OUTPUT = Path("raw_data/1-after-trim")
DEFAULT_BLACK_THRESHOLD = 8
DEFAULT_MIN_ACTIVE_FRACTION = 0.03


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"cannot decode image: {path}")
    if image.ndim not in (2, 3):
        raise RuntimeError(f"unsupported image shape {image.shape}: {path}")
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise RuntimeError(f"unsupported channel count {image.shape[2]}: {path}")
    return image


def grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[:, :, 0]
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def largest_true_run(mask: np.ndarray) -> tuple[int, int]:
    if mask.ndim != 1:
        raise ValueError("run mask must be one-dimensional")
    transitions = np.flatnonzero(
        np.diff(np.concatenate(([False], mask.astype(bool), [False])))
    ).reshape(-1, 2)
    if not len(transitions):
        raise RuntimeError("no active image region detected")
    start, stop = max(transitions, key=lambda pair: int(pair[1] - pair[0]))
    return int(start), int(stop)


def detect_crop(
    image: np.ndarray,
    black_threshold: int,
    min_active_fraction: float,
) -> dict[str, int | float]:
    gray = grayscale(image)
    non_black = gray > black_threshold
    column_fraction = non_black.mean(axis=0)
    row_fraction = non_black.mean(axis=1)
    xmin, xmax = largest_true_run(column_fraction >= min_active_fraction)
    ymin, ymax = largest_true_run(row_fraction >= min_active_fraction)
    if not (0 <= xmin < xmax <= gray.shape[1]):
        raise RuntimeError(f"invalid horizontal crop [{xmin}, {xmax})")
    if not (0 <= ymin < ymax <= gray.shape[0]):
        raise RuntimeError(f"invalid vertical crop [{ymin}, {ymax})")
    return {
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
        "width": xmax - xmin,
        "height": ymax - ymin,
        "left_active_fraction": float(column_fraction[xmin]),
        "right_active_fraction": float(column_fraction[xmax - 1]),
        "top_active_fraction": float(row_fraction[ymin]),
        "bottom_active_fraction": float(row_fraction[ymax - 1]),
    }


def source_codec(path: Path) -> str:
    with path.open("rb") as handle:
        magic = handle.read(8)
    if magic.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if magic.startswith(b"\xff\xd8"):
        return "jpeg"
    raise RuntimeError(f"unsupported source image encoding: {path}")


def write_image_atomic(path: Path, image: np.ndarray, codec: str) -> None:
    if codec == "png":
        extension = ".png"
        parameters = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    elif codec == "jpeg":
        extension = ".jpg"
        parameters = [cv2.IMWRITE_JPEG_QUALITY, 100]
    else:
        raise ValueError(f"unsupported codec: {codec}")
    ok, encoded = cv2.imencode(extension, image, parameters)
    if not ok:
        raise RuntimeError(f"cannot encode image: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        encoded.tofile(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def rounded(value: float) -> float:
    result = round(float(value), 6)
    return 0.0 if abs(result) < 0.0000005 else result


def transform_annotation(
    annotation: dict[str, Any], crop: dict[str, int | float]
) -> int:
    values = annotation.get("keypoints")
    if not isinstance(values, list) or len(values) % 3:
        raise RuntimeError(f"invalid keypoints: annotation {annotation.get('id')}")

    xmin = int(crop["xmin"])
    ymin = int(crop["ymin"])
    width = int(crop["width"])
    height = int(crop["height"])
    transformed: list[float | int] = []
    visible_points: list[tuple[float, float]] = []
    for index in range(0, len(values), 3):
        x, y, visibility = values[index : index + 3]
        x_value = float(x)
        y_value = float(y)
        visibility_value = float(visibility)
        if not all(
            math.isfinite(item)
            for item in (x_value, y_value, visibility_value)
        ):
            raise RuntimeError(
                f"non-finite keypoint: annotation {annotation.get('id')}"
            )
        if visibility_value <= 0:
            transformed.extend([x, y, visibility])
            continue
        new_x = rounded(x_value - xmin)
        new_y = rounded(y_value - ymin)
        if not (0 <= new_x < width and 0 <= new_y < height):
            raise RuntimeError(
                f"annotation {annotation.get('id')} keypoint leaves crop"
            )
        transformed.extend([new_x, new_y, visibility])
        visible_points.append((new_x, new_y))

    if not visible_points:
        raise RuntimeError(f"annotation {annotation.get('id')} has no visible points")
    box_xmin = min(point[0] for point in visible_points)
    box_ymin = min(point[1] for point in visible_points)
    box_xmax = max(point[0] for point in visible_points)
    box_ymax = max(point[1] for point in visible_points)
    bbox = [
        rounded(box_xmin),
        rounded(box_ymin),
        rounded(box_xmax - box_xmin),
        rounded(box_ymax - box_ymin),
    ]
    annotation["keypoints"] = transformed
    annotation["bbox"] = bbox
    annotation["area"] = rounded(bbox[2] * bbox[3])
    annotation["num_keypoints"] = len(visible_points)
    return len(visible_points)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_crop_csv(path: Path, records: Sequence[dict[str, Any]]) -> None:
    fieldnames = [
        "file_name",
        "source_width",
        "source_height",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "width",
        "height",
        "removed_left",
        "removed_top",
        "removed_right",
        "removed_bottom",
        "left_active_fraction",
        "right_active_fraction",
        "top_active_fraction",
        "bottom_active_fraction",
        "source_codec",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: record[key] for key in fieldnames} for record in records)


def validate_dataset(root: Path) -> dict[str, Any]:
    annotation_path = root / "annotations/person_keypoints_default.json"
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    if len(images) != 116 or len(annotations) != 812:
        raise RuntimeError(
            f"unexpected output counts: {len(images)} images, "
            f"{len(annotations)} annotations"
        )
    image_ids = [int(image["id"]) for image in images]
    file_names = [str(image["file_name"]) for image in images]
    annotation_ids = [int(annotation["id"]) for annotation in annotations]
    if len(set(image_ids)) != len(image_ids):
        raise RuntimeError("duplicate output image IDs")
    if len(set(file_names)) != len(file_names):
        raise RuntimeError("duplicate output file names")
    if len(set(annotation_ids)) != len(annotation_ids):
        raise RuntimeError("duplicate output annotation IDs")

    image_by_id = {int(image["id"]): image for image in images}
    image_dir = root / "images/default"
    disk_files = sorted(path.name for path in image_dir.iterdir() if path.is_file())
    if sorted(file_names) != disk_files:
        raise RuntimeError("output image files do not match COCO file names")

    codec_counts: Counter[str] = Counter()
    for image_info in images:
        path = image_dir / str(image_info["file_name"])
        image = decode_image(path)
        height, width = image.shape[:2]
        if (width, height) != (
            int(image_info["width"]),
            int(image_info["height"]),
        ):
            raise RuntimeError(f"COCO/image size mismatch: {path.name}")
        codec_counts[source_codec(path)] += 1

    category_counts: Counter[int] = Counter()
    visible_points = 0
    for annotation in annotations:
        image_id = int(annotation["image_id"])
        if image_id not in image_by_id:
            raise RuntimeError(f"orphan annotation: {annotation['id']}")
        image_info = image_by_id[image_id]
        width = int(image_info["width"])
        height = int(image_info["height"])
        values = annotation["keypoints"]
        current_visible = 0
        for index in range(0, len(values), 3):
            x, y, visibility = map(float, values[index : index + 3])
            if visibility > 0:
                current_visible += 1
                if not (0 <= x < width and 0 <= y < height):
                    raise RuntimeError(
                        f"output keypoint outside image: annotation {annotation['id']}"
                    )
        if current_visible != int(annotation["num_keypoints"]):
            raise RuntimeError(f"num_keypoints mismatch: {annotation['id']}")
        visible_points += current_visible
        category_counts[int(annotation["category_id"])] += 1

    return {
        "image_count": len(images),
        "annotation_count": len(annotations),
        "visible_keypoint_count": visible_points,
        "category_counts": dict(sorted(category_counts.items())),
        "codec_counts": dict(sorted(codec_counts.items())),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--black-threshold", type=int, default=DEFAULT_BLACK_THRESHOLD
    )
    parser.add_argument(
        "--min-active-fraction",
        type=float,
        default=DEFAULT_MIN_ACTIVE_FRACTION,
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write and publish the output; otherwise only inspect the crop plan",
    )
    args = parser.parse_args(argv)

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_dir():
        raise RuntimeError(f"source dataset does not exist: {source}")
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")
    if output == source or source in output.parents:
        raise RuntimeError("output must not be the source or a child of the source")
    if not 0 <= args.black_threshold <= 255:
        raise ValueError("black threshold must be in [0, 255]")
    if not 0 < args.min_active_fraction <= 1:
        raise ValueError("minimum active fraction must be in (0, 1]")

    coco_path = source / "annotations/person_keypoints_default.json"
    image_dir = source / "images/default"
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    if len(images) != 116 or len(annotations) != 812:
        raise RuntimeError(
            f"unexpected source counts: {len(images)} images, "
            f"{len(annotations)} annotations"
        )

    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    crop_by_image: dict[int, dict[str, int | float]] = {}
    crop_records: list[dict[str, Any]] = []
    for image_info in sorted(images, key=lambda item: str(item["file_name"])):
        image_id = int(image_info["id"])
        file_name = str(image_info["file_name"])
        path = image_dir / file_name
        image = decode_image(path)
        height, width = image.shape[:2]
        if (width, height) != (
            int(image_info["width"]),
            int(image_info["height"]),
        ):
            raise RuntimeError(f"source COCO/image size mismatch: {file_name}")
        crop = detect_crop(
            image,
            black_threshold=args.black_threshold,
            min_active_fraction=args.min_active_fraction,
        )
        crop_by_image[image_id] = crop
        record: dict[str, Any] = {
            "file_name": file_name,
            "source_width": width,
            "source_height": height,
            **crop,
            "removed_left": int(crop["xmin"]),
            "removed_top": int(crop["ymin"]),
            "removed_right": width - int(crop["xmax"]),
            "removed_bottom": height - int(crop["ymax"]),
            "source_codec": source_codec(path),
        }
        crop_records.append(record)

        for annotation in annotations_by_image[image_id]:
            values = annotation.get("keypoints", [])
            for index in range(0, len(values), 3):
                x, y, visibility = map(float, values[index : index + 3])
                if visibility <= 0:
                    continue
                if not (
                    int(crop["xmin"]) <= x < int(crop["xmax"])
                    and int(crop["ymin"]) <= y < int(crop["ymax"])
                ):
                    raise RuntimeError(
                        f"keypoint would be cropped: {file_name}, "
                        f"annotation {annotation['id']}"
                    )

    widths = [int(record["width"]) for record in crop_records]
    heights = [int(record["height"]) for record in crop_records]
    plan = {
        "source": str(source),
        "output": str(output),
        "image_count": len(images),
        "annotation_count": len(annotations),
        "black_threshold": args.black_threshold,
        "min_active_fraction": args.min_active_fraction,
        "cropped_width_range": [min(widths), max(widths)],
        "cropped_height_range": [min(heights), max(heights)],
        "images_with_vertical_crop": sum(
            int(record["removed_top"]) > 0 or int(record["removed_bottom"]) > 0
            for record in crop_records
        ),
    }
    if not args.apply:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    staging = output.with_name(f".{output.name}.staging-{os.getpid()}")
    if staging.exists():
        raise RuntimeError(f"staging path already exists: {staging}")
    try:
        (staging / "images/default").mkdir(parents=True)
        (staging / "annotations").mkdir(parents=True)

        for image_info in images:
            image_id = int(image_info["id"])
            file_name = str(image_info["file_name"])
            source_path = image_dir / file_name
            image = decode_image(source_path)
            crop = crop_by_image[image_id]
            xmin = int(crop["xmin"])
            ymin = int(crop["ymin"])
            xmax = int(crop["xmax"])
            ymax = int(crop["ymax"])
            cropped = image[ymin:ymax, xmin:xmax].copy()
            write_image_atomic(
                staging / "images/default" / file_name,
                cropped,
                source_codec(source_path),
            )
            image_info["width"] = int(crop["width"])
            image_info["height"] = int(crop["height"])

        visible_points = 0
        for annotation in annotations:
            visible_points += transform_annotation(
                annotation, crop_by_image[int(annotation["image_id"])]
            )

        output_coco_path = staging / "annotations/person_keypoints_default.json"
        write_json(output_coco_path, coco)
        write_crop_csv(staging / "crop_bounds.csv", crop_records)
        validation = validate_dataset(staging)
        if validation["visible_keypoint_count"] != visible_points:
            raise RuntimeError("visible keypoint count changed during validation")
        report = {
            **plan,
            "status": "completed",
            "source_annotation_sha256": sha256(coco_path),
            "output_annotation_sha256": sha256(output_coco_path),
            "validation": validation,
        }
        write_json(staging / "trim_report.json", report)
        os.replace(staging, output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
