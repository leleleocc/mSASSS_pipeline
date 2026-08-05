#!/usr/bin/env python3
"""Convert BUU LA CSV annotations to LabelMe JSON with estimated T12 and S1 points.

Reads the full BUU lateral (LA) dataset from CSV+JPG pairs, computes L1-L5
centroids from the four corners, estimates T12 bottom and S1 centroid using
adjacent-gap-copy + side-edge-extension, and writes a LabelMe-format JSON
per case into a train/val/test split structure matching keypoints_with_t12_s1.

CSV format per file (11 rows, no header):
  x_left, y_left, x_right, y_right, mSASSS_label
Rows map to:
  0: L1 top edge      5: L3 bottom edge
  1: L1 bottom edge   6: L4 top edge
  2: L2 top edge      7: L4 bottom edge
  3: L2 bottom edge   8: L5 top edge
  4: L3 top edge      9: L5 bottom edge
                      10: S1 top edge
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import struct
from pathlib import Path
from typing import Any

import numpy as np


LEVELS = ["L1", "L2", "L3", "L4", "L5"]
DESCRIPTION_DERIVED = "derived from BUU-LSPINE LA annotation"
DESCRIPTION_ESTIMATED = "estimated by adjacent-gap copy and side-edge extension"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--la-dir",
        type=Path,
        default=Path("/paddle/lv/mSASSS_pipeline/tmp/dataset\u0e25\u0e07\u0e2b\u0e19\u0e49\u0e32\u0e40\u0e27\u0e47\u0e1a/LA"),
        help="Directory containing CSV+JPG pairs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/paddle/lv/mSASSS_pipeline/data/raw_data/BUU-LSPINE/keypoints_with_t12_s1"),
        help="Output directory with train/val/test structure.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Remove existing output before writing.",
    )
    return parser.parse_args()


def jpeg_dimensions(path: Path) -> tuple[int, int]:
    """Read width and height from JPEG SOF marker without loading pixels."""
    with path.open("rb") as f:
        f.read(2)  # SOI
        while True:
            marker = f.read(2)
            if not marker or marker[0] != 0xFF:
                raise RuntimeError(f"cannot parse JPEG: {path}")
            if marker[1] in (0xC0, 0xC2):
                f.read(3)  # length + precision
                h = struct.unpack(">H", f.read(2))[0]
                w = struct.unpack(">H", f.read(2))[0]
                return w, h
            else:
                length = struct.unpack(">H", f.read(2))[0]
                f.read(length - 2)


def read_la_csv(path: Path) -> list[list[float]]:
    """Read 11-row LA CSV: each row is [x_left, y_left, x_right, y_right, label]."""
    rows = []
    with path.open() as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 5:
                raise ValueError(f"expected 5 columns in {path}, got {len(parts)}")
            rows.append([float(x) for x in parts])
    if len(rows) != 11:
        raise ValueError(f"expected 11 rows in {path}, got {len(rows)}")
    return rows


def parse_csv_to_corners(rows: list[list[float]]) -> dict[str, np.ndarray]:
    """Convert 11 CSV rows into named corner points for L1-L5 and S1 top.

    If left.x > right.x in any row, the pair is swapped to enforce the
    convention that 'left' always has a smaller x coordinate.
    """
    points: dict[str, np.ndarray] = {}
    for i, level in enumerate(LEVELS):
        top_row = rows[i * 2]
        bot_row = rows[i * 2 + 1]
        tl = np.array([top_row[0], top_row[1]])
        tr = np.array([top_row[2], top_row[3]])
        bl = np.array([bot_row[0], bot_row[1]])
        br = np.array([bot_row[2], bot_row[3]])
        if tl[0] > tr[0]:
            tl, tr = tr, tl
        if bl[0] > br[0]:
            bl, br = br, bl
        points[f"{level} top left"] = tl
        points[f"{level} top right"] = tr
        points[f"{level} bottom left"] = bl
        points[f"{level} bottom right"] = br
    # Row 10: S1 top edge
    s1_tl = np.array([rows[10][0], rows[10][1]])
    s1_tr = np.array([rows[10][2], rows[10][3]])
    if s1_tl[0] > s1_tr[0]:
        s1_tl, s1_tr = s1_tr, s1_tl
    points["S1 top left"] = s1_tl
    points["S1 top right"] = s1_tr
    return points


def compute_centroids(points: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute L1-L5 centroids as mean of four corners."""
    centroids: dict[str, np.ndarray] = {}
    for level in LEVELS:
        corners = np.array([
            points[f"{level} top left"],
            points[f"{level} top right"],
            points[f"{level} bottom left"],
            points[f"{level} bottom right"],
        ])
        centroids[f"{level} centroid"] = corners.mean(axis=0)
    return centroids


def _reflect_across_line(
    point: np.ndarray, line_p1: np.ndarray, line_p2: np.ndarray
) -> np.ndarray:
    """Reflect a point across a line defined by two points."""
    d = line_p2 - line_p1
    d = d / np.linalg.norm(d)
    v = point - line_p1
    proj = np.dot(v, d) * d
    perp = v - proj
    return point - 2 * perp


def estimate_t12_s1(
    points: dict[str, np.ndarray], centroids: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Estimate T12 bottom corners/centroid and S1 centroid.

    Method: adjacent-gap copy + side-edge extension (same as original pipeline).
    - T12 bottom left  = L1 top left  - (L2 top left  - L1 bottom left)
    - T12 bottom right = L1 top right - (L2 top right - L1 bottom right)
    - T12 centroid     = L1 centroid  + (L1 centroid  - L2 centroid)
    - S1 centroid      = L5 centroid  + (L5 centroid  - L4 centroid)
    """
    estimated: dict[str, np.ndarray] = {}

    # T12 bottom: copy disc gap between L1 and L2 above L1
    gap_left = points["L2 top left"] - points["L1 bottom left"]
    gap_right = points["L2 top right"] - points["L1 bottom right"]
    estimated["T12 bottom left"] = points["L1 top left"] - gap_left
    estimated["T12 bottom right"] = points["L1 top right"] - gap_right

    # T12 centroid: extend from L1 centroid
    estimated["T12 centroid"] = (
        centroids["L1 centroid"]
        + (centroids["L1 centroid"] - centroids["L2 centroid"])
    )

    # S1 centroid: reflect L5 centroid across the L5-S1 disc center line.
    # Disc center line = line connecting mid(S1_TL, L5_BL) to mid(S1_TR, L5_BR)
    mid_left = (points["S1 top left"] + points["L5 bottom left"]) / 2
    mid_right = (points["S1 top right"] + points["L5 bottom right"]) / 2
    estimated["S1 centroid"] = _reflect_across_line(
        centroids["L5 centroid"], mid_left, mid_right
    )

    return estimated


def point_is_outside(point: np.ndarray, width: int, height: int) -> bool:
    x, y = float(point[0]), float(point[1])
    return not (0.0 <= x < width and 0.0 <= y < height)


def make_shape(
    label: str,
    point: np.ndarray,
    group_id: str | None,
    *,
    estimated: bool = False,
    width: int = 0,
    height: int = 0,
) -> dict[str, Any]:
    """Create a LabelMe point shape."""
    flags: dict[str, bool] = {"derived": True}
    if estimated:
        flags["estimated"] = True
        flags["outside_image"] = point_is_outside(point, width, height)
        flags["visible"] = True
        desc = DESCRIPTION_ESTIMATED
    else:
        desc = DESCRIPTION_DERIVED
    return {
        "label": label,
        "points": [[float(point[0]), float(point[1])]],
        "group_id": group_id,
        "description": desc,
        "shape_type": "point",
        "flags": flags,
    }


def build_labelme_json(
    case_name: str,
    image_name: str,
    width: int,
    height: int,
    points: dict[str, np.ndarray],
    centroids: dict[str, np.ndarray],
    estimated: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Build a complete LabelMe annotation dict."""
    shapes: list[dict[str, Any]] = []

    # L1-L5 corners and centroids
    for level in LEVELS:
        for pos in ("top left", "top right", "bottom left", "bottom right"):
            shapes.append(make_shape(
                f"{level} {pos}", points[f"{level} {pos}"], level
            ))
        shapes.append(make_shape(
            f"{level} centroid", centroids[f"{level} centroid"], level
        ))

    # T12 estimated points
    for label in ("T12 bottom left", "T12 bottom right", "T12 centroid"):
        shapes.append(make_shape(
            label, estimated[label], "T12",
            estimated=True, width=width, height=height,
        ))

    # S1 top (directly from annotation)
    shapes.append(make_shape("S1 top left", points["S1 top left"], "S1"))
    shapes.append(make_shape("S1 top right", points["S1 top right"], "S1"))

    # S1 centroid (estimated)
    shapes.append(make_shape(
        "S1 centroid", estimated["S1 centroid"], "S1",
        estimated=True, width=width, height=height,
    ))

    return {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }


def split_cases(
    cases: list[str],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, str]:
    """Assign each case to train/val/test."""
    shuffled = sorted(cases)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_test = max(1, round(n * test_fraction))
    n_val = max(1, round(n * val_fraction))
    mapping: dict[str, str] = {}
    for case in shuffled[:n_test]:
        mapping[case] = "test"
    for case in shuffled[n_test : n_test + n_val]:
        mapping[case] = "val"
    for case in shuffled[n_test + n_val :]:
        mapping[case] = "train"
    return mapping


def main() -> int:
    args = parse_args()
    la_dir = args.la_dir.resolve()
    output = args.output.resolve()

    if not la_dir.is_dir():
        raise FileNotFoundError(f"LA directory not found: {la_dir}")

    # Collect all cases (stem without extension)
    csv_files = sorted(la_dir.glob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"no CSV files in {la_dir}")

    cases = [p.stem for p in csv_files]
    print(f"Found {len(cases)} cases in {la_dir}")

    # Verify each case has a matching JPG
    for case in cases:
        jpg = la_dir / f"{case}.jpg"
        if not jpg.exists():
            raise FileNotFoundError(f"missing image for {case}: {jpg}")

    # Assign splits
    split_map = split_cases(cases, args.val_fraction, args.test_fraction, args.seed)

    # Handle output directory
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"output exists: {output}. Use --overwrite to replace."
            )
        shutil.rmtree(output)

    # Create output structure
    for split in ("train", "val", "test"):
        (output / split / "labels").mkdir(parents=True)
        (output / split / "images").mkdir(parents=True)

    # Process each case
    outside_cases: list[dict[str, Any]] = []
    counts = {"train": 0, "val": 0, "test": 0, "estimated_points": 0}
    errors: list[dict[str, str]] = []

    for i, case in enumerate(cases, 1):
        try:
            csv_path = la_dir / f"{case}.csv"
            jpg_path = la_dir / f"{case}.jpg"
            split = split_map[case]

            rows = read_la_csv(csv_path)
            width, height = jpeg_dimensions(jpg_path)
            corners = parse_csv_to_corners(rows)
            centroids = compute_centroids(corners)
            estimated = estimate_t12_s1(corners, centroids)

            annotation = build_labelme_json(
                case, f"{case}.jpg", width, height,
                corners, centroids, estimated,
            )

            # Check for outside-image estimated points
            outside = []
            for label, pt in estimated.items():
                if point_is_outside(pt, width, height):
                    outside.append({"label": label, "point": [float(pt[0]), float(pt[1])]})
            if outside:
                outside_cases.append({
                    "case": case, "split": split,
                    "image_size": [width, height], "outside": outside,
                })

            # Write outputs
            out_json = output / split / "labels" / f"{case}.json"
            out_json.write_text(
                json.dumps(annotation, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            out_img = output / split / "images" / f"{case}.jpg"
            os.link(jpg_path, out_img)

            counts[split] += 1
            counts["estimated_points"] += len(estimated)

        except Exception as e:
            errors.append({"case": case, "error": str(e)})

        if i % 500 == 0 or i == len(cases):
            print(f"  processed {i}/{len(cases)}")

    # Write report
    report = {
        "input_root": str(la_dir),
        "output_root": str(output),
        "counts": counts,
        "total_cases": len(cases),
        "outside_image_cases": outside_cases,
        "errors": errors,
    }
    (output / "extension_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(counts, indent=2))
    print(f"Outside-image cases: {len(outside_cases)}")
    if errors:
        print(f"ERRORS: {len(errors)}")
        for e in errors[:10]:
            print(f"  {e['case']}: {e['error']}")
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
