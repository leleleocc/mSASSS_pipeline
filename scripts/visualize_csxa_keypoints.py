#!/usr/bin/env python3
"""Visualize one CSXA LabelMe annotation together with its paired image."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np


LEVEL_COLORS = {
    "C2": (0, 255, 255),
    "C3": (0, 165, 255),
    "C4": (0, 255, 0),
    "C5": (255, 200, 0),
    "C6": (255, 0, 255),
    "C7": (0, 80, 255),
    "T1": (255, 255, 0),
}
POSITION_ABBREVIATIONS = {
    "top left": "TL",
    "top right": "TR",
    "bottom left": "BL",
    "bottom right": "BR",
    "centroid": "CTR",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("raw_data/CSXA/origin/datasets-PNG"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("raw_data/CSXA/visualizations/csxa_keypoints.jpg"),
    )
    return parser.parse_args()


def parse_label(label: str) -> tuple[str, str]:
    normalized = " ".join(label.strip().split())
    match = re.match(r"^(C[2-7]|T1)\s+(.+)$", normalized, flags=re.IGNORECASE)
    if not match:
        return "unknown", normalized
    return match.group(1).upper(), match.group(2).casefold()


def load_points(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for shape in annotation.get("shapes", []):
        points = shape.get("points", [])
        if not points or any(len(point) != 2 for point in points):
            continue
        point_array = np.asarray(points, dtype=np.float64)
        xy = np.median(point_array, axis=0)
        # A few CSXA landmarks were accidentally saved as 2-4 almost-identical
        # polygon vertices. Collapse only these tight clusters; reject real
        # multi-point polygons rather than silently treating them as landmarks.
        if len(points) > 1 and np.max(np.linalg.norm(point_array - xy, axis=1)) > 2.0:
            continue
        level, position = parse_label(str(shape.get("label", "")))
        records.append(
            {
                "label": str(shape.get("label", "")),
                "level": level,
                "position": position,
                "xy": xy,
            }
        )
    return records


def draw_annotation(image: np.ndarray, records: list[dict[str, Any]], title: str) -> np.ndarray:
    canvas = image.copy()
    height, width = canvas.shape[:2]
    radius = max(4, round(min(height, width) / 150))
    lookup = {(record["level"], record["position"]): record["xy"] for record in records}

    for level in LEVEL_COLORS:
        color = LEVEL_COLORS[level]
        corners = [
            lookup.get((level, "top left")),
            lookup.get((level, "top right")),
            lookup.get((level, "bottom right")),
            lookup.get((level, "bottom left")),
        ]
        if all(point is not None for point in corners):
            polygon = np.rint(np.asarray(corners)).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [polygon], True, color, 2, cv2.LINE_AA)
        else:
            top_left, top_right, bottom_left, bottom_right = (
                lookup.get((level, "top left")),
                lookup.get((level, "top right")),
                lookup.get((level, "bottom left")),
                lookup.get((level, "bottom right")),
            )
            for left, right in ((top_left, top_right), (bottom_left, bottom_right)):
                if left is not None and right is not None:
                    cv2.line(
                        canvas,
                        tuple(np.rint(left).astype(int)),
                        tuple(np.rint(right).astype(int)),
                        color,
                        2,
                        cv2.LINE_AA,
                    )

    for record in records:
        color = LEVEL_COLORS.get(record["level"], (255, 255, 255))
        x, y = np.rint(record["xy"]).astype(int)
        cv2.circle(canvas, (x, y), radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), radius, color, -1, cv2.LINE_AA)
        abbreviation = POSITION_ABBREVIATIONS.get(record["position"], record["position"][:3].upper())
        text = f"{record['level']}-{abbreviation}"
        cv2.putText(
            canvas,
            text,
            (x + radius + 3, y - radius - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            text,
            (x + radius + 3, y - radius - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    bar_height = 74
    cv2.rectangle(canvas, (0, 0), (width, bar_height), (0, 0, 0), -1)
    cv2.putText(canvas, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    legend_x = 12
    for level, color in LEVEL_COLORS.items():
        cv2.circle(canvas, (legend_x + 7, 55), 6, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, level, (legend_x + 18, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
        legend_x += 70
    return canvas


def main() -> int:
    args = parse_args()
    json_path = args.json.resolve()
    annotation = json.loads(json_path.read_text(encoding="utf-8"))
    image_name = Path(str(annotation.get("imagePath") or f"{json_path.stem}.png")).name
    image_path = args.image_dir.resolve() / image_name
    if not image_path.is_file():
        image_path = args.image_dir.resolve() / f"{json_path.stem}.png"
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read paired image: {image_path}")
    records = load_points(annotation)
    if not records:
        raise RuntimeError(f"no single-point annotations found in {json_path}")
    title = f"CSXA {json_path.stem} | {len(records)} keypoints | {image.shape[1]}x{image.shape[0]}"
    visualization = draw_annotation(image, records, title)
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), visualization):
        raise RuntimeError(f"failed to write {output_path}")
    print(f"image: {image_path}")
    print(f"annotation: {json_path}")
    print(f"keypoints: {len(records)}")
    print(f"visualization: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
