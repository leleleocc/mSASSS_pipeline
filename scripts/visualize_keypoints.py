#!/usr/bin/env python3
"""Render COCO spine keypoints for selected images."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


def decode_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"cannot decode {path}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError(f"cannot encode {path}")
    encoded.tofile(path)


def points(annotation: dict[str, Any]) -> list[tuple[float, float]]:
    values = annotation["keypoints"]
    result = []
    for index in range(0, len(values), 3):
        x, y, visibility = map(float, values[index : index + 3])
        if visibility <= 0:
            raise RuntimeError(f"invisible point in annotation {annotation['id']}")
        result.append((x, y))
    return result


def order_vu_annotations(
    bone_points: Sequence[tuple[float, float]],
    annotations: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    interval_midpoints = [
        (
            (bone_points[index][0] + bone_points[index + 1][0]) / 2,
            (bone_points[index][1] + bone_points[index + 1][1]) / 2,
        )
        for index in range(6)
    ]
    assigned: dict[int, dict[str, Any]] = {}
    for annotation in annotations:
        pair = points(annotation)
        midpoint = (
            (pair[0][0] + pair[1][0]) / 2,
            (pair[0][1] + pair[1][1]) / 2,
        )
        interval = min(
            range(6),
            key=lambda index: (
                (midpoint[0] - interval_midpoints[index][0]) ** 2
                + (midpoint[1] - interval_midpoints[index][1]) ** 2
            ),
        )
        if interval in assigned:
            raise RuntimeError(f"duplicate VU interval assignment: {interval}")
        assigned[interval] = annotation
    if set(assigned) != set(range(6)):
        raise RuntimeError(f"incomplete VU assignments: {sorted(assigned)}")
    return [assigned[index] for index in range(6)]


def render(
    image: np.ndarray,
    file_name: str,
    bone_annotation: dict[str, Any],
    vu_annotations: Sequence[dict[str, Any]],
) -> np.ndarray:
    height, width = image.shape[:2]
    scale = max(0.55, min(1.15, min(width, height) / 1100))
    radius = max(5, round(8 * scale))
    thickness = max(2, round(3 * scale))
    font_scale = max(0.48, 0.66 * scale)
    bone_points = points(bone_annotation)
    ordered_vus = order_vu_annotations(bone_points, vu_annotations)

    canvas = image.copy()
    bone_pixels = [(round(x), round(y)) for x, y in bone_points]
    cv2.polylines(
        canvas,
        [np.asarray(bone_pixels, dtype=np.int32)],
        False,
        (0, 0, 255),
        thickness,
        cv2.LINE_AA,
    )
    for index, center in enumerate(bone_pixels, start=1):
        cv2.circle(canvas, center, radius + 2, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, center, radius, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"B{index}",
            (center[0] + radius + 4, center[1] - radius - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    for index, annotation in enumerate(ordered_vus, start=1):
        pair = [(round(x), round(y)) for x, y in points(annotation)]
        cv2.line(canvas, pair[0], pair[1], (255, 255, 0), thickness, cv2.LINE_AA)
        cv2.circle(canvas, pair[0], radius + 1, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, pair[0], radius - 1, (255, 180, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, pair[1], radius + 1, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, pair[1], radius - 1, (0, 255, 255), -1, cv2.LINE_AA)
        midpoint = (
            (pair[0][0] + pair[1][0]) // 2,
            (pair[0][1] + pair[1][1]) // 2,
        )
        cv2.putText(
            canvas,
            f"V{index}",
            (midpoint[0] - round(48 * scale), midpoint[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    header_height = max(58, round(72 * scale))
    result = cv2.copyMakeBorder(
        canvas, header_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(18, 18, 18)
    )
    title = (
        f"{file_name} | {width}x{height} | "
        "red=B1-B7, cyan=P1, yellow=P2"
    )
    cv2.putText(
        result,
        title,
        (12, round(header_height * 0.67)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        max(1, thickness - 1),
        cv2.LINE_AA,
    )
    return result


def make_contact_sheet(images: Sequence[np.ndarray]) -> np.ndarray:
    target_height = 1000
    resized = []
    for image in images:
        scale = target_height / image.shape[0]
        resized.append(
            cv2.resize(
                image,
                (round(image.shape[1] * scale), target_height),
                interpolation=cv2.INTER_AREA,
            )
        )
    gap = 24
    canvas = np.full(
        (target_height, sum(image.shape[1] for image in resized) + gap, 3),
        24,
        np.uint8,
    )
    x = 0
    for image in resized:
        canvas[:, x : x + image.shape[1]] = image
        x += image.shape[1] + gap
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    args = parser.parse_args()

    root = args.dataset.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    coco = json.loads(
        (root / "annotations/person_keypoints_default.json").read_text(
            encoding="utf-8"
        )
    )
    categories = {
        int(category["id"]): str(category["name"]).casefold()
        for category in coco["categories"]
    }
    image_by_name = {str(image["file_name"]): image for image in coco["images"]}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    rendered = []
    for file_name in args.cases:
        if file_name not in image_by_name:
            raise RuntimeError(f"unknown case: {file_name}")
        image_info = image_by_name[file_name]
        annotations = annotations_by_image[int(image_info["id"])]
        bone = [a for a in annotations if categories[int(a["category_id"])] == "bone"]
        vus = [a for a in annotations if categories[int(a["category_id"])] == "vu"]
        if len(bone) != 1 or len(vus) != 6:
            raise RuntimeError(
                f"unexpected annotations for {file_name}: bone={len(bone)}, vu={len(vus)}"
            )
        image = decode_image(root / "images/default" / file_name)
        visualization = render(image, file_name, bone[0], vus)
        output_path = output / f"{Path(file_name).stem}_keypoints.png"
        write_png(output_path, visualization)
        rendered.append(visualization)

    write_png(output / "two_cases_contact_sheet.png", make_contact_sheet(rendered))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
