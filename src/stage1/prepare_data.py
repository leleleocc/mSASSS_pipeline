#!/usr/bin/env python3
"""Build one 19-point spine landmark dataset from Zhongri, CSXA and BUU.

Unified point order:
  B1..B7, V1P1, V1P2, ..., V6P1, V6P2

B1..B7 are seven vertebral centers.  Each V pair is the left inferior corner
of the superior vertebra followed by the left superior corner of the inferior
vertebra.  Cervical and lumbar images share this topology while retaining a
class ID that selects their anatomical name mapping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE1_ROOT = Path(__file__).resolve().parent
MPL_CONFIG = STAGE1_ROOT / ".matplotlib"
MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

KEYPOINT_NAMES = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "V1P1",
    "V1P2",
    "V2P1",
    "V2P2",
    "V3P1",
    "V3P2",
    "V4P1",
    "V4P2",
    "V5P1",
    "V5P2",
    "V6P1",
    "V6P2",
]
CLASS_NAMES = {0: "cervical_spine", 1: "lumbar_spine"}
CERVICAL_NAMES = [
    "C2 centroid",
    "C3 centroid",
    "C4 centroid",
    "C5 centroid",
    "C6 centroid",
    "C7 centroid",
    "T1 centroid",
    "C2 bottom left",
    "C3 top left",
    "C3 bottom left",
    "C4 top left",
    "C4 bottom left",
    "C5 top left",
    "C5 bottom left",
    "C6 top left",
    "C6 bottom left",
    "C7 top left",
    "C7 bottom left",
    "T1 top left",
]
LUMBAR_NAMES = [
    "T12 centroid",
    "L1 centroid",
    "L2 centroid",
    "L3 centroid",
    "L4 centroid",
    "L5 centroid",
    "S1 centroid",
    "T12 bottom left",
    "L1 top left",
    "L1 bottom left",
    "L2 top left",
    "L2 bottom left",
    "L3 top left",
    "L3 bottom left",
    "L4 top left",
    "L4 bottom left",
    "L5 top left",
    "L5 bottom left",
    "S1 top left",
]
ANATOMICAL_NAMES = {0: CERVICAL_NAMES, 1: LUMBAR_NAMES}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zhongri",
        type=Path,
        default=PROJECT_ROOT / "raw_data/zhongri/1-after-trim",
    )
    parser.add_argument(
        "--csxa",
        type=Path,
        default=PROJECT_ROOT / "raw_data/CSXA/CSXA-trans",
    )
    parser.add_argument(
        "--buu",
        type=Path,
        default=PROJECT_ROOT / "raw_data/BUU-LSPINE/keypoints_with_t12_s1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=STAGE1_ROOT / "data/spine_keypoints_19pt",
    )
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace an existing generated output dataset.",
    )
    return parser.parse_args()


def normalized_label(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def labelme_points(annotation: dict[str, Any]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for shape in annotation.get("shapes", []):
        label = normalized_label(str(shape.get("label", "")))
        points = shape.get("points", [])
        if len(points) != 1 or len(points[0]) != 2:
            continue
        if label in result:
            raise ValueError(f"duplicate point label: {label}")
        result[label] = np.asarray(points[0], dtype=np.float64)
    return result


def get_point(lookup: dict[str, np.ndarray], label: str) -> np.ndarray:
    key = normalized_label(label)
    if key not in lookup:
        raise ValueError(f"missing point: {label}")
    return lookup[key]


def four_corner_center(lookup: dict[str, np.ndarray], level: str) -> np.ndarray:
    return np.mean(
        [
            get_point(lookup, f"{level} top left"),
            get_point(lookup, f"{level} top right"),
            get_point(lookup, f"{level} bottom left"),
            get_point(lookup, f"{level} bottom right"),
        ],
        axis=0,
    )


def visible_points(annotation: dict[str, Any]) -> list[np.ndarray]:
    values = annotation.get("keypoints")
    if not isinstance(values, list) or len(values) % 3:
        raise ValueError(f"invalid COCO keypoints in annotation {annotation.get('id')}")
    result: list[np.ndarray] = []
    for index in range(0, len(values), 3):
        x, y, visibility = values[index : index + 3]
        if int(visibility) <= 0:
            raise ValueError(f"invisible COCO point in annotation {annotation.get('id')}")
        result.append(np.asarray([float(x), float(y)], dtype=np.float64))
    return result


def order_zhongri_vus(
    centers: Sequence[np.ndarray], vu_annotations: Sequence[dict[str, Any]]
) -> tuple[list[np.ndarray], int]:
    interval_midpoints = [
        (centers[index] + centers[index + 1]) / 2.0 for index in range(6)
    ]
    assigned: dict[int, list[np.ndarray]] = {}
    swaps = 0
    for annotation in vu_annotations:
        pair = visible_points(annotation)
        if len(pair) != 2:
            raise ValueError(f"VU annotation must contain 2 points: {annotation['id']}")
        midpoint = (pair[0] + pair[1]) / 2.0
        interval = min(
            range(6),
            key=lambda index: float(
                np.sum((midpoint - interval_midpoints[index]) ** 2)
            ),
        )
        if interval in assigned:
            raise ValueError(f"duplicate VU assignment at interval {interval + 1}")
        direction = centers[interval + 1] - centers[interval]
        projections = [float(np.dot(point - centers[interval], direction)) for point in pair]
        ordered = [point for _, point in sorted(zip(projections, pair), key=lambda item: item[0])]
        swaps += int(not np.array_equal(ordered[0], pair[0]))
        assigned[interval] = ordered
    if set(assigned) != set(range(6)):
        raise ValueError(f"incomplete VU assignments: {sorted(assigned)}")
    flattened = [point for index in range(6) for point in assigned[index]]
    return flattened, swaps


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.width, image.height


def validate_points(
    points: np.ndarray, width: int, height: int
) -> list[dict[str, Any]]:
    if points.shape != (19, 2) or not np.isfinite(points).all():
        raise ValueError(f"invalid point array: shape={points.shape}")
    outside: list[dict[str, Any]] = []
    for index, (x, y) in enumerate(points):
        if not (0 <= x < width and 0 <= y < height):
            outside.append(
                {
                    "index": index,
                    "generic_name": KEYPOINT_NAMES[index],
                    "point": [float(x), float(y)],
                }
            )
    return outside


def record(
    *,
    source: str,
    source_case: str,
    group_id: str,
    class_id: int,
    image_path: Path,
    width: int,
    height: int,
    points: Sequence[np.ndarray],
    native_split: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    point_array = np.asarray(points, dtype=np.float64)
    actual_width, actual_height = image_size(image_path)
    if (width, height) != (actual_width, actual_height):
        raise ValueError(
            f"dimension mismatch for {image_path}: annotation {(width, height)}, "
            f"image {(actual_width, actual_height)}"
        )
    return {
        "source": source,
        "source_case": source_case,
        "group_id": f"{source}:{group_id}",
        "class_id": class_id,
        "image_path": image_path,
        "width": width,
        "height": height,
        "points": point_array,
        "outside": validate_points(point_array, width, height),
        "native_split": native_split,
        "metadata": metadata or {},
    }


def collect_zhongri(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    annotation_path = root / "annotations" / "person_keypoints_default.json"
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    category_names = {
        int(item["id"]): str(item["name"]).casefold()
        for item in coco.get("categories", [])
    }
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco.get("annotations", []):
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    records: list[dict[str, Any]] = []
    groups: set[str] = set()
    for item in coco.get("images", []):
        file_name = str(item["file_name"])
        stem = Path(file_name).stem
        if "-" not in stem:
            raise ValueError(f"cannot parse Zhongri patient/view: {file_name}")
        patient, view = stem.rsplit("-", 1)
        if view not in {"C", "L"}:
            raise ValueError(f"unknown Zhongri view: {file_name}")
        class_id = 0 if view == "C" else 1
        groups.add(patient)
        annotations = annotations_by_image[int(item["id"])]
        bones = [
            annotation
            for annotation in annotations
            if category_names[int(annotation["category_id"])] == "bone"
        ]
        vus = [
            annotation
            for annotation in annotations
            if category_names[int(annotation["category_id"])] == "vu"
        ]
        if len(bones) != 1 or len(vus) != 6:
            raise ValueError(f"{file_name}: expected one bone and six VUs")
        centers = visible_points(bones[0])
        if len(centers) != 7:
            raise ValueError(f"{file_name}: expected seven centers")
        corners, swaps = order_zhongri_vus(centers, vus)
        records.append(
            record(
                source="zhongri",
                source_case=stem,
                group_id=patient,
                class_id=class_id,
                image_path=root / "images" / "default" / file_name,
                width=int(item["width"]),
                height=int(item["height"]),
                points=[*centers, *corners],
                metadata={"view": view, "pair_order_swaps": swaps},
            )
        )
    return records, sorted(groups)


def collect_csxa(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    groups: set[str] = set()
    json_dir = root / "datasets-JSON"
    image_dir = root / "datasets-PNG"
    for path in sorted(json_dir.glob("*.json")):
        annotation = json.loads(path.read_text(encoding="utf-8"))
        lookup = labelme_points(annotation)
        centers = [get_point(lookup, "C2 centroid")]
        centers.extend(four_corner_center(lookup, f"C{level}") for level in range(3, 8))
        centers.append(get_point(lookup, "T1 centroid"))
        corners: list[np.ndarray] = [get_point(lookup, "C2 bottom left")]
        for level in range(3, 8):
            corners.extend(
                [
                    get_point(lookup, f"C{level} top left"),
                    get_point(lookup, f"C{level} bottom left"),
                ]
            )
        corners.append(get_point(lookup, "T1 top left"))
        # Reorder the alternating chain into six [upper BL, lower TL] pairs.
        paired_corners = [
            corners[index] for index in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
        ]
        image_name = Path(str(annotation.get("imagePath") or f"{path.stem}.png")).name
        group_id = path.stem[:4]
        groups.add(group_id)
        records.append(
            record(
                source="csxa",
                source_case=path.stem,
                group_id=group_id,
                class_id=0,
                image_path=image_dir / image_name,
                width=int(annotation["imageWidth"]),
                height=int(annotation["imageHeight"]),
                points=[*centers, *paired_corners],
                metadata={"T1_points_estimated": True},
            )
        )
    return records, sorted(groups)


def collect_buu(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    groups: set[str] = set()
    for split in ("train", "val", "test"):
        label_dir = root / split / "labels"
        image_dir = root / split / "images"
        for path in sorted(label_dir.glob("*.json")):
            annotation = json.loads(path.read_text(encoding="utf-8"))
            lookup = labelme_points(annotation)
            centers = [get_point(lookup, "T12 centroid")]
            centers.extend(get_point(lookup, f"L{level} centroid") for level in range(1, 6))
            centers.append(get_point(lookup, "S1 centroid"))
            corners: list[np.ndarray] = [get_point(lookup, "T12 bottom left")]
            for level in range(1, 6):
                corners.extend(
                    [
                        get_point(lookup, f"L{level} top left"),
                        get_point(lookup, f"L{level} bottom left"),
                    ]
                )
            corners.append(get_point(lookup, "S1 top left"))
            image_name = Path(str(annotation["imagePath"])).name
            groups.add(path.stem)
            records.append(
                record(
                    source="buu",
                    source_case=path.stem,
                    group_id=path.stem,
                    class_id=1,
                    image_path=image_dir / image_name,
                    width=int(annotation["imageWidth"]),
                    height=int(annotation["imageHeight"]),
                    points=[*centers, *corners],
                    native_split=None,
                    metadata={"T12_and_S1_points_estimated": True},
                )
            )
    return records, sorted(groups)


def split_groups(
    groups: Sequence[str], val_fraction: float, test_fraction: float, seed: int
) -> dict[str, str]:
    shuffled = sorted(set(groups))
    random.Random(seed).shuffle(shuffled)
    test_count = max(1, round(len(shuffled) * test_fraction))
    val_count = max(1, round(len(shuffled) * val_fraction))
    if test_count + val_count >= len(shuffled):
        raise ValueError("not enough groups for requested split fractions")
    result: dict[str, str] = {}
    for group in shuffled[:test_count]:
        result[group] = "test"
    for group in shuffled[test_count : test_count + val_count]:
        result[group] = "val"
    for group in shuffled[test_count + val_count :]:
        result[group] = "train"
    return result


def assign_splits(
    records: list[dict[str, Any]],
    source_groups: dict[str, list[str]],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> None:
    mappings = {
        "zhongri": split_groups(
            source_groups["zhongri"], val_fraction, test_fraction, seed + 101
        ),
        "csxa": split_groups(
            source_groups["csxa"], val_fraction, test_fraction, seed + 202
        ),
        "buu": split_groups(
            source_groups["buu"], val_fraction, test_fraction, seed + 303
        ),
    }
    for item in records:
        raw_group = str(item["group_id"]).split(":", 1)[1]
        item["split"] = mappings[item["source"]][raw_group]


def yolo_line(item: dict[str, Any]) -> str:
    width = int(item["width"])
    height = int(item["height"])
    points = item["points"]
    xs = points[:, 0]
    ys = points[:, 1]
    xmin = max(0.0, float(xs.min()) - 0.10 * width)
    xmax = min(float(width), float(xs.max()) + 0.10 * width)
    ymin = max(0.0, float(ys.min()) - 0.03 * height)
    ymax = min(float(height), float(ys.max()) + 0.03 * height)
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"invalid pose box: {item['source_case']}")
    fields = [
        str(item["class_id"]),
        f"{((xmin + xmax) / 2) / width:.8f}",
        f"{((ymin + ymax) / 2) / height:.8f}",
        f"{(xmax - xmin) / width:.8f}",
        f"{(ymax - ymin) / height:.8f}",
    ]
    for x, y in points:
        fields.extend([f"{x / width:.8f}", f"{y / height:.8f}", "2"])
    return " ".join(fields)


def output_name(item: dict[str, Any]) -> str:
    suffix = item["image_path"].suffix.casefold()
    return f"{item['source']}__{item['source_case']}{suffix}"


def link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def write_data_yaml(path: Path, final_root: Path) -> None:
    identity = ", ".join(str(index) for index in range(19))
    content = (
        f'path: "{final_root}"\n'
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "kpt_shape: [19, 3]\n"
        f"flip_idx: [{identity}]\n\n"
        "names:\n"
        "  0: cervical_spine\n"
        "  1: lumbar_spine\n"
    )
    path.write_text(content, encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_duplicate_groups(
    records: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    by_size: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        by_size[item["image_path"].stat().st_size].append(item)
    duplicates: list[list[dict[str, Any]]] = []
    for same_size in by_size.values():
        if len(same_size) < 2:
            continue
        by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in same_size:
            by_hash[file_sha256(item["image_path"])].append(item)
        for digest, same_hash in by_hash.items():
            if len(same_hash) > 1:
                for item in same_hash:
                    item["image_sha256"] = digest
                duplicates.append(same_hash)
    return duplicates


def deduplicate_exact_images(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep one deterministic record for each byte-identical image.

    Manual Zhongri labels are preferred, followed by CSXA and BUU derived
    labels.  Ties use the lexicographically first source case.  This avoids
    overweighting one radiograph when a public mirror repeats it under two IDs.
    """
    source_priority = {"zhongri": 0, "csxa": 1, "buu": 2}
    groups = exact_duplicate_groups(records)
    dropped_ids: set[int] = set()
    audit: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for group in groups:
        ordered = sorted(
            group,
            key=lambda item: (
                source_priority[item["source"]],
                item["source"],
                item["source_case"],
            ),
        )
        retained = ordered[0]
        dropped = ordered[1:]
        dropped_ids.update(id(item) for item in dropped)
        audit.append(
            {
                "sha256": retained["image_sha256"],
                "retained": {
                    "source": retained["source"],
                    "case": retained["source_case"],
                    "split": retained["split"],
                },
                "dropped": [
                    {
                        "source": item["source"],
                        "case": item["source_case"],
                        "split": item["split"],
                    }
                    for item in dropped
                ],
                "original_splits": sorted({item["split"] for item in group}),
            }
        )
        for item in dropped:
            exclusions.append(
                {
                    "source": item["source"],
                    "source_case": item["source_case"],
                    "class_id": item["class_id"],
                    "native_or_assigned_split": item["split"],
                    "reason": "byte-identical duplicate image",
                    "retained": {
                        "source": retained["source"],
                        "source_case": retained["source_case"],
                        "split": retained["split"],
                    },
                }
            )
    kept = [item for item in records if id(item) not in dropped_ids]
    return kept, audit, exclusions


def draw_samples(records: list[dict[str, Any]], path: Path, seed: int) -> None:
    buckets: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        buckets[(item["source"], item["class_id"])].append(item)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for key in sorted(buckets):
        values = buckets[key]
        selected.extend(rng.sample(values, min(2, len(values))))
    columns = 4
    rows = int(np.ceil(len(selected) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(16, 6 * rows))
    axes_array = np.asarray(axes, dtype=object).reshape(-1)
    for axis, item in zip(axes_array, selected):
        image = Image.open(item["image_path"])
        axis.imshow(image, cmap="gray" if image.mode == "L" else None)
        points = item["points"]
        axis.plot(points[:7, 0], points[:7, 1], "-o", color="#ffcc00", linewidth=1.3, markersize=3)
        for index, point in enumerate(points[:7], start=1):
            axis.text(point[0] + 4, point[1] - 4, f"B{index}", color="#ffcc00", fontsize=6)
        for interval in range(6):
            first = points[7 + interval * 2]
            second = points[8 + interval * 2]
            axis.plot(
                [first[0], second[0]],
                [first[1], second[1]],
                "-o",
                color="#00e5ff",
                linewidth=1.5,
                markersize=3,
            )
            axis.text(first[0] + 4, first[1] - 3, f"V{interval + 1}P1", color="#00e5ff", fontsize=5)
            axis.text(second[0] + 4, second[1] + 8, f"V{interval + 1}P2", color="#00e5ff", fontsize=5)
        axis.set_title(
            f"{item['source']} | {CLASS_NAMES[item['class_id']]} | {item['split']}\n{item['source_case']}",
            fontsize=8,
        )
        axis.set_xlim(0, item["width"])
        axis.set_ylim(item["height"], 0)
        axis.axis("off")
    for axis in axes_array[len(selected) :]:
        axis.axis("off")
    figure.suptitle(
        "Unified 19-point dataset | yellow: 7 centers | cyan: 6 left-corner pairs",
        fontsize=14,
        weight="bold",
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.val_fraction <= 0 or args.test_fraction <= 0:
        raise ValueError("validation and test fractions must be positive")
    if args.val_fraction + args.test_fraction >= 1:
        raise ValueError("validation plus test fractions must be less than one")

    roots = {
        "zhongri": args.zhongri.resolve(),
        "csxa": args.csxa.resolve(),
        "buu": args.buu.resolve(),
    }
    for name, root in roots.items():
        if not root.is_dir():
            raise FileNotFoundError(f"{name} root does not exist: {root}")

    zhongri, zhongri_groups = collect_zhongri(roots["zhongri"])
    csxa, csxa_groups = collect_csxa(roots["csxa"])
    buu, buu_groups = collect_buu(roots["buu"])
    all_records = [*zhongri, *csxa, *buu]
    assign_splits(
        all_records,
        {
            "zhongri": zhongri_groups,
            "csxa": csxa_groups,
            "buu": buu_groups,
        },
        args.val_fraction,
        args.test_fraction,
        args.seed,
    )

    excluded: list[dict[str, Any]] = []
    included: list[dict[str, Any]] = []
    for item in all_records:
        if item["outside"]:
            excluded.append(
                {
                    "source": item["source"],
                    "source_case": item["source_case"],
                    "class_id": item["class_id"],
                    "native_or_assigned_split": item["split"],
                    "reason": "one or more required keypoints outside image",
                    "outside": item["outside"],
                }
            )
        else:
            included.append(item)

    included, duplicate_groups, duplicate_exclusions = deduplicate_exact_images(
        included
    )
    excluded.extend(duplicate_exclusions)

    counts = Counter(
        (item["source"], item["split"], CLASS_NAMES[item["class_id"]])
        for item in included
    )
    summary = {
        "output": str(args.output.resolve()),
        "seed": args.seed,
        "split_policy": {
            "zhongri": "80/10/10 patient-wise; paired C/L stay together",
            "csxa": "80/10/10 by unique four-digit case prefix",
            "buu": "preserve source train/val/test split",
        },
        "keypoints_per_image": 19,
        "keypoint_names": KEYPOINT_NAMES,
        "classes": {str(key): value for key, value in CLASS_NAMES.items()},
        "source_records_before_filter": {
            "zhongri": len(zhongri),
            "csxa": len(csxa),
            "buu": len(buu),
        },
        "included_images": len(included),
        "excluded_images": len(excluded),
        "excluded_by_reason": dict(
            sorted(Counter(item["reason"] for item in excluded).items())
        ),
        "exact_duplicate_groups": len(duplicate_groups),
        "duplicate_images_removed": len(duplicate_exclusions),
        "included_counts": [
            {
                "source": source,
                "split": split,
                "class": class_name,
                "images": count,
            }
            for (source, split, class_name), count in sorted(counts.items())
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.check_only:
        return 0

    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {output}")
    staging = output.with_name(f".{output.name}.staging-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"staging path already exists: {staging}")

    try:
        for split in ("train", "val", "test"):
            (staging / "images" / split).mkdir(parents=True)
            (staging / "labels" / split).mkdir(parents=True)
        (staging / "visualizations").mkdir(parents=True)

        transfer_modes: Counter[str] = Counter()
        manifest: list[dict[str, Any]] = []
        seen_output_names: set[str] = set()
        for item in sorted(included, key=lambda value: (value["split"], output_name(value))):
            name = output_name(item)
            if name in seen_output_names:
                raise ValueError(f"duplicate output name: {name}")
            seen_output_names.add(name)
            split = item["split"]
            target_image = staging / "images" / split / name
            transfer_modes[link_or_copy(item["image_path"], target_image)] += 1
            target_label = staging / "labels" / split / f"{Path(name).stem}.txt"
            target_label.write_text(yolo_line(item) + "\n", encoding="utf-8")
            manifest.append(
                {
                    "file_name": name,
                    "label_name": target_label.name,
                    "split": split,
                    "source": item["source"],
                    "source_case": item["source_case"],
                    "group_id": item["group_id"],
                    "class_id": item["class_id"],
                    "class_name": CLASS_NAMES[item["class_id"]],
                    "width": item["width"],
                    "height": item["height"],
                    "metadata": item["metadata"],
                }
            )

        summary["image_transfer"] = dict(transfer_modes)
        summary["cross_split_exact_duplicate_groups_before_deduplication"] = sum(
            len(item["original_splits"]) > 1 for item in duplicate_groups
        )

        write_data_yaml(staging / "data.yaml", output)
        (staging / "keypoint_schema.json").write_text(
            json.dumps(
                {
                    "generic_order": KEYPOINT_NAMES,
                    "class_anatomical_order": {
                        CLASS_NAMES[class_id]: ANATOMICAL_NAMES[class_id]
                        for class_id in CLASS_NAMES
                    },
                    "pair_semantics": (
                        "VxP1 = superior vertebra left bottom corner; "
                        "VxP2 = inferior vertebra left top corner"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (staging / "excluded.json").write_text(
            json.dumps(excluded, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (staging / "duplicate_audit.json").write_text(
            json.dumps(duplicate_groups, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (staging / "dataset_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        draw_samples(
            included,
            staging / "visualizations" / "unified_keypoint_samples.png",
            args.seed,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            backup = output.with_name(f".{output.name}.backup-{os.getpid()}")
            os.replace(output, backup)
            try:
                os.replace(staging, output)
            except Exception:
                os.replace(backup, output)
                raise
            shutil.rmtree(backup)
        else:
            os.replace(staging, output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    print(f"dataset: {output}")
    print(f"visualization: {output / 'visualizations' / 'unified_keypoint_samples.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
