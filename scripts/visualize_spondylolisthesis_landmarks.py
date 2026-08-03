#!/usr/bin/env python3
"""Render BUU spondylolisthesis vertebral boxes and four-corner landmarks."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path


COLORS = (
    (255, 64, 64),
    (64, 255, 96),
    (64, 160, 255),
    (255, 224, 64),
    (255, 64, 224),
    (64, 255, 255),
    (255, 144, 32),
    (160, 96, 255),
    (160, 255, 64),
    (255, 128, 192),
    (128, 255, 192),
)


@dataclass(frozen=True)
class Instance:
    box: tuple[float, float, float, float]
    points: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class Sample:
    image: Path
    label: Path
    split: str


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def samples(root: Path) -> list[Sample]:
    base = root / "Train/Keypointrcnn_data"
    result: list[Sample] = []
    for split in ("train", "val"):
        image_dir = base / "images" / split
        for label in sorted((base / "labels" / split).glob("*")):
            if label.suffix.lower() not in {".json", ".txt"}:
                continue
            image = image_dir / f"{label.stem}.jpg"
            if image.is_file():
                result.append(Sample(image=image, label=label, split=split))
    return result


def dimensions(image: Path) -> tuple[int, int]:
    value = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=s=x:p=0",
            str(image),
        ],
        text=True,
    ).strip()
    width, height = value.split("x")
    return int(width), int(height)


def json_instances(label: Path) -> list[Instance]:
    annotation = json.loads(label.read_text(encoding="utf-8"))
    boxes = annotation.get("boxes", [])
    keypoints = annotation.get("keypoints", [])
    if len(boxes) != len(keypoints):
        raise ValueError(f"box/keypoint count mismatch: {label}")
    return [
        Instance(
            box=tuple(float(value) for value in box),
            points=tuple((float(point[0]), float(point[1])) for point in points),
        )
        for box, points in zip(boxes, keypoints)
    ]


def yolo_instances(label: Path, width: int, height: int) -> list[Instance]:
    result: list[Instance] = []
    for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        if len(values) != 17:
            raise ValueError(f"expected 17 YOLO-Pose fields at {label}:{line_number}")
        _, cx, cy, box_width, box_height, *keypoints = values
        box = (
            (cx - box_width / 2) * width,
            (cy - box_height / 2) * height,
            (cx + box_width / 2) * width,
            (cy + box_height / 2) * height,
        )
        raw_points = tuple(
            (keypoints[index] * width, keypoints[index + 1] * height)
            for index in range(0, len(keypoints), 3)
        )
        # YOLO files store TL, BL, BR, TR; normalize to the JSON order
        # BL, BR, TL, TR before rendering.
        points = tuple(raw_points[index] for index in (1, 2, 0, 3))
        result.append(Instance(box=box, points=points))
    return result


def set_pixel(buffer: bytearray, width: int, height: int, x: int, y: int, color: tuple[int, int, int]) -> None:
    if 0 <= x < width and 0 <= y < height:
        offset = (y * width + x) * 3
        buffer[offset : offset + 3] = bytes(color)


def circle(buffer: bytearray, width: int, height: int, x: float, y: float, radius: int, color: tuple[int, int, int]) -> None:
    center_x, center_y = round(x), round(y)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                set_pixel(buffer, width, height, center_x + dx, center_y + dy, color)


def line(buffer: bytearray, width: int, height: int, start: tuple[float, float], end: tuple[float, float], color: tuple[int, int, int], thickness: int = 2) -> None:
    x0, y0 = round(start[0]), round(start[1])
    x1, y1 = round(end[0]), round(end[1])
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    step_x, step_y = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    error = dx + dy
    while True:
        circle(buffer, width, height, x0, y0, thickness, color)
        if x0 == x1 and y0 == y1:
            return
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += step_x
        if doubled <= dx:
            error += dx
            y0 += step_y


def rectangle(buffer: bytearray, width: int, height: int, box: tuple[float, float, float, float], color: tuple[int, int, int]) -> None:
    left, top, right, bottom = box
    corners = ((left, top), (right, top), (right, bottom), (left, bottom))
    for index in range(4):
        line(buffer, width, height, corners[index], corners[(index + 1) % 4], color, 1)


def render(sample: Sample, output: Path) -> None:
    width, height = dimensions(sample.image)
    pixels = bytearray(
        subprocess.check_output(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(sample.image),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ]
        )
    )
    instances = (
        json_instances(sample.label)
        if sample.label.suffix.lower() == ".json"
        else yolo_instances(sample.label, width, height)
    )
    instances.sort(key=lambda item: sum(point[1] for point in item.points) / len(item.points))
    for index, instance in enumerate(instances):
        color = COLORS[index % len(COLORS)]
        if len(instance.points) != 4:
            raise ValueError(f"expected four points per vertebra: {sample.label}")
        rectangle(pixels, width, height, instance.box, color)
        # Stored order: bottom-left, bottom-right, top-left, top-right.
        perimeter = (2, 3, 1, 0, 2)
        for first, second in zip(perimeter, perimeter[1:]):
            line(pixels, width, height, instance.points[first], instance.points[second], color, 2)
        for point in instance.points:
            circle(pixels, width, height, point[0], point[1], 6, color)
            circle(pixels, width, height, point[0], point[1], 2, (255, 255, 255))

    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-i",
            "pipe:0",
            "-frames:v",
            "1",
            str(output),
        ],
        input=pixels,
        check=True,
    )


def choose(items: list[Sample], count: int, seed: int) -> list[Sample]:
    rng = random.Random(seed)
    by_format = {
        suffix: [item for item in items if item.label.suffix.lower() == suffix]
        for suffix in (".json", ".txt")
    }
    for values in by_format.values():
        rng.shuffle(values)
    json_count = (count + 1) // 2
    chosen = by_format[".json"][:json_count] + by_format[".txt"][: count - json_count]
    rng.shuffle(chosen)
    return chosen


def contact_sheet(images: list[Path], output: Path) -> None:
    columns = 3
    rows = (len(images) + columns - 1) // columns
    inputs: list[str] = []
    filters: list[str] = []
    labels: list[str] = []
    for index, path in enumerate(images):
        inputs.extend(("-i", str(path)))
        label = f"v{index}"
        filters.append(f"[{index}:v]scale=480:480[{label}]")
        labels.append(f"[{label}]")
    layout = "|".join(f"{(index % columns) * 480}_{(index // columns) * 480}" for index in range(len(images)))
    filters.append(f"{''.join(labels)}xstack=inputs={len(images)}:layout={layout}:fill=black[out]")
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            *inputs,
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            "-frames:v",
            "1",
            str(output),
        ],
        check=True,
    )


def main() -> int:
    args = arguments()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    available = samples(args.dataset.resolve())
    selected = choose(available, min(args.count, len(available)), args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    rendered: list[Path] = []
    manifest: list[dict[str, str]] = []
    for index, sample in enumerate(selected, 1):
        output = args.output / f"{index:02d}_{sample.split}_{sample.label.suffix[1:]}_{sample.label.stem}.png"
        render(sample, output)
        rendered.append(output)
        manifest.append(
            {
                "cell": str(index),
                "split": sample.split,
                "format": sample.label.suffix[1:],
                "image": str(sample.image),
                "label": str(sample.label),
                "visualization": str(output),
            }
        )
    contact_sheet(rendered, args.output / "overview.png")
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
