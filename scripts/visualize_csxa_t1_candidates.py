#!/usr/bin/env python3
"""Review estimated T1 landmarks on random CSXA images without changing annotations."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from visualize_csxa_keypoints import draw_annotation, load_points


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
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("raw_data/CSXA/visualizations/t1_gap_copy_review"),
    )
    return parser.parse_args()


def point_lookup(records: list[dict[str, Any]]) -> dict[tuple[str, str], np.ndarray]:
    return {(record["level"], record["position"]): record["xy"] for record in records}


def project_to_line(
    point: np.ndarray,
    line_start: np.ndarray,
    line_end: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return the orthogonal projection and fractional position on an infinite line."""
    direction = line_end - line_start
    squared_length = float(np.dot(direction, direction))
    if squared_length < 1e-8:
        raise RuntimeError("cannot project onto a zero-length line")
    fraction = float(np.dot(point - line_start, direction) / squared_length)
    projection = line_start + fraction * direction
    return projection, fraction


def estimate_t1(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Extrapolate review-only T1 landmarks by copying C6/C7 geometry.

    The two intervertebral gap lengths are copied independently, while the
    directions follow the extensions of the C7 left and right edges:

        length(C6_BL, C7_TL) == length(C7_BL, T1_TL)
        length(C6_BR, C7_TR) == length(C7_BR, T1_TR)

    The C7 center is derived from its four corners. Its perpendicular distance
    and fractional projection position relative to the C7 top edge are then
    transferred to the estimated T1 top edge.
    """
    lookup = point_lookup(records)
    positions = ("top left", "top right", "bottom left", "bottom right")
    required = [(level, position) for level in ("C6", "C7") for position in positions]
    missing = [key for key in required if key not in lookup]
    if missing:
        raise RuntimeError(f"missing C6/C7 landmarks: {missing}")
    c7_center = np.mean([lookup[("C7", position)] for position in positions], axis=0)

    # Copy each C6-bottom -> C7-top gap length along the corresponding C7
    # side-edge extension. The source vector direction itself is not copied.
    c7_top_left = lookup[("C7", "top left")]
    c7_top_right = lookup[("C7", "top right")]
    c7_bottom_left = lookup[("C7", "bottom left")]
    c7_bottom_right = lookup[("C7", "bottom right")]
    left_gap = np.linalg.norm(c7_top_left - lookup[("C6", "bottom left")])
    right_gap = np.linalg.norm(c7_top_right - lookup[("C6", "bottom right")])
    left_edge_direction = c7_bottom_left - c7_top_left
    right_edge_direction = c7_bottom_right - c7_top_right
    left_edge_direction /= np.linalg.norm(left_edge_direction)
    right_edge_direction /= np.linalg.norm(right_edge_direction)
    t1_top_left = c7_bottom_left + left_gap * left_edge_direction
    t1_top_right = c7_bottom_right + right_gap * right_edge_direction

    # Copy the C7 center-to-top-edge perpendicular geometry onto the T1 top edge.
    c7_projection, lateral_fraction = project_to_line(
        c7_center, c7_top_left, c7_top_right
    )
    c7_top_direction = c7_top_right - c7_top_left
    c7_down_normal = np.asarray(
        [-c7_top_direction[1], c7_top_direction[0]], dtype=np.float64
    )
    c7_down_normal /= np.linalg.norm(c7_down_normal)
    signed_distance = float(np.dot(c7_center - c7_projection, c7_down_normal))

    t1_projection = t1_top_left + lateral_fraction * (t1_top_right - t1_top_left)
    t1_top_direction = t1_top_right - t1_top_left
    t1_down_normal = np.asarray(
        [-t1_top_direction[1], t1_top_direction[0]], dtype=np.float64
    )
    t1_down_normal /= np.linalg.norm(t1_down_normal)
    t1_center = t1_projection + signed_distance * t1_down_normal
    return {
        "T1-TL?": t1_top_left,
        "T1-TR?": t1_top_right,
        "T1-CTR?": t1_center,
    }


def draw_candidates(
    image: np.ndarray,
    records: list[dict[str, Any]],
    candidates: dict[str, np.ndarray],
) -> np.ndarray:
    canvas = image.copy()
    lookup = point_lookup(records)
    color = (255, 255, 0)
    connections = (
        (lookup[("C7", "bottom left")], candidates["T1-TL?"]),
        (lookup[("C7", "bottom right")], candidates["T1-TR?"]),
    )
    for start, end in connections:
        cv2.line(
            canvas,
            tuple(np.rint(start).astype(int)),
            tuple(np.rint(end).astype(int)),
            color,
            2,
            cv2.LINE_AA,
        )
    for label, point in candidates.items():
        x, y = np.rint(point).astype(int)
        cv2.drawMarker(
            canvas,
            (x, y),
            color,
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=18,
            thickness=3,
            line_type=cv2.LINE_AA,
        )
        cv2.putText(canvas, label, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    height, width = canvas.shape[:2]
    cv2.rectangle(canvas, (0, height - 30), (width, height), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        "cyan X = extrapolated T1 candidate (review only, not ground truth)",
        (8, height - 9),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def draw_clean_t1_overlay(
    image: np.ndarray,
    records: list[dict[str, Any]],
    candidates: dict[str, np.ndarray],
) -> np.ndarray:
    """Draw the source geometry and copied T1 geometry without other labels."""
    canvas = image.copy()
    lookup = point_lookup(records)
    corner_order = ("top left", "top right", "bottom right", "bottom left")
    level_colors = {"C6": (255, 0, 255), "C7": (0, 80, 255)}
    for level, color in level_colors.items():
        polygon = np.rint(
            [lookup[(level, position)] for position in corner_order]
        ).astype(np.int32)
        cv2.polylines(
            canvas, [polygon.reshape(-1, 1, 2)], True, color, 3, cv2.LINE_AA
        )
        for position in corner_order:
            xy = tuple(np.rint(lookup[(level, position)]).astype(int))
            cv2.circle(canvas, xy, 5, color, -1, cv2.LINE_AA)

    # Yellow arrows provide the source lengths; cyan arrows follow C7 side edges.
    arrow_pairs = (
        (
            lookup[("C6", "bottom left")],
            lookup[("C7", "top left")],
            (0, 255, 255),
        ),
        (
            lookup[("C6", "bottom right")],
            lookup[("C7", "top right")],
            (0, 255, 255),
        ),
        (
            lookup[("C7", "bottom left")],
            candidates["T1-TL?"],
            (255, 255, 0),
        ),
        (
            lookup[("C7", "bottom right")],
            candidates["T1-TR?"],
            (255, 255, 0),
        ),
    )
    for start, end, color in arrow_pairs:
        cv2.arrowedLine(
            canvas,
            tuple(np.rint(start).astype(int)),
            tuple(np.rint(end).astype(int)),
            color,
            3,
            cv2.LINE_AA,
            tipLength=0.22,
        )

    c7_center = np.mean(
        [lookup[("C7", position)] for position in corner_order], axis=0
    )
    c7_projection, _ = project_to_line(
        c7_center,
        lookup[("C7", "top left")],
        lookup[("C7", "top right")],
    )
    t1_projection, _ = project_to_line(
        candidates["T1-CTR?"], candidates["T1-TL?"], candidates["T1-TR?"]
    )
    cv2.line(
        canvas,
        tuple(np.rint(candidates["T1-TL?"]).astype(int)),
        tuple(np.rint(candidates["T1-TR?"]).astype(int)),
        (0, 255, 255),
        3,
        cv2.LINE_AA,
    )
    for projection, center in (
        (c7_projection, c7_center),
        (t1_projection, candidates["T1-CTR?"]),
    ):
        cv2.line(
            canvas,
            tuple(np.rint(projection).astype(int)),
            tuple(np.rint(center).astype(int)),
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.circle(
            canvas,
            tuple(np.rint(center).astype(int)),
            6,
            (0, 255, 0),
            -1,
            cv2.LINE_AA,
        )

    candidate_styles = {
        "T1-TL?": ((255, 255, 0), "T1 TL"),
        "T1-TR?": ((255, 255, 0), "T1 TR"),
        "T1-CTR?": ((0, 255, 0), "T1 center"),
    }
    for key, point in candidates.items():
        color, label = candidate_styles[key]
        xy = tuple(np.rint(point).astype(int))
        cv2.circle(canvas, xy, 7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.circle(canvas, xy, 6, color, -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            label,
            (xy[0] + 8, xy[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            (xy[0] + 8, xy[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas


def add_geometry_legend(image: np.ndarray) -> np.ndarray:
    """Add a readable legend after cropping so it remains visible."""
    bar_height = 76
    canvas = np.full((image.shape[0] + bar_height, image.shape[1], 3), 16, np.uint8)
    canvas[: image.shape[0]] = image
    lines = (
        ("yellow = C6 BL/BR -> C7 TL/TR", (0, 255, 255)),
        ("cyan = same lengths along C7 side edges", (255, 255, 0)),
        ("green = equal center-to-top distance", (0, 255, 0)),
    )
    for index, (label, color) in enumerate(lines):
        cv2.putText(
            canvas,
            label,
            (8, image.shape[0] + 20 + index * 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas


def raw_vs_overlay(raw: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    target_height, target_width = 390, 300

    def fit(image: np.ndarray) -> np.ndarray:
        scale = min(target_width / image.shape[1], target_height / image.shape[0])
        width = max(1, round(image.shape[1] * scale))
        height = max(1, round(image.shape[0] * scale))
        tile = np.full((target_height, target_width, 3), 18, np.uint8)
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        x = (target_width - width) // 2
        y = (target_height - height) // 2
        tile[y : y + height, x : x + width] = resized
        return tile

    combined = np.concatenate([fit(raw), fit(overlay)], axis=1)
    cv2.putText(combined, "RAW", (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        combined,
        "C6/C7 geometry + T1",
        (target_width + 8, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return combined


def crop_lower_cervical(
    image: np.ndarray,
    records: list[dict[str, Any]],
    candidates: dict[str, np.ndarray],
) -> np.ndarray:
    lookup = point_lookup(records)
    points = [
        point
        for (level, _), point in lookup.items()
        if level in {"C6", "C7"}
    ] + list(candidates.values())
    array = np.asarray(points)
    height, width = image.shape[:2]
    span_x = max(float(np.ptp(array[:, 0])), 60.0)
    span_y = max(float(np.ptp(array[:, 1])), 100.0)
    x1 = max(0, int(np.floor(array[:, 0].min() - max(90.0, span_x * 1.2))))
    x2 = min(width, int(np.ceil(array[:, 0].max() + max(90.0, span_x * 1.2))))
    y1 = max(0, int(np.floor(array[:, 1].min() - max(50.0, span_y * 0.45))))
    y2 = min(height, int(np.ceil(array[:, 1].max() + max(120.0, span_y * 0.9))))
    return image[y1:y2, x1:x2].copy()


def contact_sheet(
    images: list[tuple[str, np.ndarray]],
    output_path: Path,
    columns: int,
    tile_width: int,
    tile_height: int,
) -> None:
    rows = (len(images) + columns - 1) // columns
    canvas = np.full((rows * tile_height, columns * tile_width, 3), 22, np.uint8)
    for index, (name, image) in enumerate(images):
        available_height = tile_height - 34
        scale = min(tile_width / image.shape[1], available_height / image.shape[0])
        width = max(1, round(image.shape[1] * scale))
        height = max(1, round(image.shape[0] * scale))
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        tile_x = (index % columns) * tile_width
        tile_y = (index // columns) * tile_height
        x = tile_x + (tile_width - width) // 2
        y = tile_y + 34 + (available_height - height) // 2
        canvas[y : y + height, x : x + width] = resized
        cv2.putText(canvas, name, (tile_x + 8, tile_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"failed to write {output_path}")


def main() -> int:
    args = parse_args()
    json_paths = sorted(args.json_dir.resolve().glob("*.json"))
    if not json_paths:
        raise RuntimeError(f"no JSON files in {args.json_dir}")
    rng = random.Random(args.seed)
    selected = rng.sample(json_paths, min(args.count, len(json_paths)))
    output_dir = args.output.resolve()
    individual_dir = output_dir / "individual"
    individual_dir.mkdir(parents=True, exist_ok=True)
    full_images: list[tuple[str, np.ndarray]] = []
    zoom_images: list[tuple[str, np.ndarray]] = []
    comparison_images: list[tuple[str, np.ndarray]] = []
    geometry_demo_images: list[tuple[str, np.ndarray]] = []

    for rank, json_path in enumerate(selected, start=1):
        annotation = json.loads(json_path.read_text(encoding="utf-8"))
        image_path = args.image_dir.resolve() / f"{json_path.stem}.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read {image_path}")
        records = load_points(annotation)
        candidates = estimate_t1(records)
        title = f"CSXA {json_path.stem} | existing labels + extrapolated T1 review"
        annotated = draw_annotation(image, records, title)
        annotated = draw_candidates(annotated, records, candidates)
        zoom = crop_lower_cervical(annotated, records, candidates)
        raw_zoom = crop_lower_cervical(image, records, candidates)
        clean_overlay = draw_clean_t1_overlay(image, records, candidates)
        clean_zoom = crop_lower_cervical(clean_overlay, records, candidates)
        full_path = individual_dir / f"{rank:02d}_{json_path.stem}_full.jpg"
        zoom_path = individual_dir / f"{rank:02d}_{json_path.stem}_t1_zoom.jpg"
        geometry_path = individual_dir / f"{rank:02d}_{json_path.stem}_geometry.jpg"
        cv2.imwrite(str(full_path), annotated)
        cv2.imwrite(str(zoom_path), zoom)
        cv2.imwrite(str(geometry_path), add_geometry_legend(clean_zoom))
        full_images.append((json_path.stem, annotated))
        zoom_images.append((json_path.stem, zoom))
        comparison_images.append((json_path.stem, raw_vs_overlay(raw_zoom, clean_zoom)))
        if rank <= 6:
            geometry_demo_images.append(
                (json_path.stem, add_geometry_legend(clean_zoom))
            )

    contact_sheet(full_images, output_dir / "full_contact_sheet.jpg", 3, 460, 610)
    contact_sheet(zoom_images, output_dir / "t1_zoom_contact_sheet.jpg", 4, 420, 420)
    contact_sheet(
        comparison_images,
        output_dir / "t1_raw_vs_candidate_contact_sheet.jpg",
        3,
        620,
        460,
    )
    contact_sheet(
        geometry_demo_images,
        output_dir / "geometry_demo_contact_sheet.jpg",
        3,
        540,
        520,
    )
    print(f"samples: {len(selected)}")
    print(f"full overview: {output_dir / 'full_contact_sheet.jpg'}")
    print(f"T1 zoom overview: {output_dir / 't1_zoom_contact_sheet.jpg'}")
    print(
        "raw vs candidate: "
        f"{output_dir / 't1_raw_vs_candidate_contact_sheet.jpg'}"
    )
    print(f"geometry demo: {output_dir / 'geometry_demo_contact_sheet.jpg'}")
    print(f"individual images: {individual_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
