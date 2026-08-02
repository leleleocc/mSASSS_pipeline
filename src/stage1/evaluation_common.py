"""Model-independent Stage-1 point metrics, tables, plots, and overlays."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


KEYPOINT_NAMES = (
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
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class CasePrediction:
    """One original-image prediction in a model-independent representation."""

    image_path: Path
    gt_class: int
    gt_points: np.ndarray
    pred_class: int | None
    pred_points: np.ndarray | None
    confidence: float | None = None


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (RuntimeError, ValueError):
            pass
    return value


def resolve_split(
    data_yaml: Path,
    split: str,
) -> tuple[dict[str, Any], list[Path], Path, Path]:
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    dataset_root = Path(config.get("path", data_yaml.parent))
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml.parent / dataset_root).resolve()
    split_value = config.get(split)
    if not isinstance(split_value, str):
        raise RuntimeError(f"{split!r} must be a directory path in {data_yaml}")
    image_dir = Path(split_value)
    if not image_dir.is_absolute():
        image_dir = dataset_root / image_dir
    label_dir = dataset_root / "labels" / image_dir.name
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise RuntimeError(f"missing image or label directory for split {split!r}")
    image_paths = sorted(
        path for path in image_dir.iterdir() if path.suffix.casefold() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise RuntimeError(f"no images found for split {split!r}")
    return config, image_paths, label_dir, dataset_root


def read_pose_label(
    label_path: Path,
    width: int,
    height: int,
) -> tuple[int, np.ndarray]:
    lines = [
        line
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) != 1:
        raise RuntimeError(f"expected one pose instance in {label_path}, found {len(lines)}")
    values = np.asarray([float(value) for value in lines[0].split()], dtype=np.float64)
    expected = 5 + len(KEYPOINT_NAMES) * 3
    if len(values) != expected:
        raise RuntimeError(f"expected {expected} fields in {label_path}, found {len(values)}")
    class_id = int(values[0])
    keypoints = values[5:].reshape(len(KEYPOINT_NAMES), 3)
    if not np.all(keypoints[:, 2] > 0):
        raise RuntimeError(f"all keypoints must be visible in {label_path}")
    points = keypoints[:, :2].copy()
    points[:, 0] *= width
    points[:, 1] *= height
    return class_id, points


def class_name(names: Any, class_id: int | None) -> str:
    if class_id is None:
        return "missing"
    if isinstance(names, dict):
        return str(names.get(class_id, names.get(str(class_id), class_id)))
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def load_manifest(dataset_root: Path) -> dict[str, dict[str, Any]]:
    path = dataset_root / "manifest.json"
    if not path.is_file():
        return {}
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise RuntimeError(f"manifest must contain a list: {path}")
    return {
        str(record["file_name"]): record
        for record in records
        if isinstance(record, dict) and "file_name" in record
    }


def _metric_block(errors: np.ndarray, normalized: np.ndarray) -> dict[str, float]:
    return {
        "mean_error_px": float(errors.mean()),
        "median_error_px": float(np.median(errors)),
        "p95_error_px": float(np.percentile(errors, 95)),
        "max_error_px": float(errors.max()),
        "mean_error_image_diag_pct": float(normalized.mean() * 100),
        "pck_0.5pct_image_diag": float((normalized <= 0.005).mean()),
        "pck_1pct_image_diag": float((normalized <= 0.01).mean()),
        "pck_2pct_image_diag": float((normalized <= 0.02).mean()),
    }


def _draw_case(record: dict[str, Any], output_path: Path) -> None:
    image = cv2.imread(str(record["image_path"]), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {record['image_path']}")
    gt = record["gt_points"]
    predicted = record["pred_points"]
    radius = max(3, round(min(image.shape[:2]) / 320))
    for index, target_point in enumerate(gt, start=1):
        target_xy = tuple(np.rint(target_point).astype(int))
        cv2.circle(image, target_xy, radius + 1, (0, 255, 0), -1, cv2.LINE_AA)
        if predicted is not None:
            predicted_xy = tuple(np.rint(predicted[index - 1]).astype(int))
            cv2.line(
                image,
                target_xy,
                predicted_xy,
                (0, 255, 255),
                max(1, radius // 2),
                cv2.LINE_AA,
            )
            cv2.circle(image, predicted_xy, radius, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(
                image,
                str(index),
                (predicted_xy[0] + radius + 1, predicted_xy[1] - radius - 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.35, radius / 8),
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    title = (
        f"{record['image_path'].name} | GT={record['gt_name']} "
        f"Pred={record['pred_name']} | MRE={record['mean_error_px']:.1f}px "
        f"N-MRE={record['mean_error_image_diag_pct']:.3f}%"
    )
    legend = "GT green | prediction red | error yellow | labels 1-19"
    bar_height = max(66, round(image.shape[0] * 0.065))
    cv2.rectangle(image, (0, 0), (image.shape[1], bar_height), (0, 0, 0), -1)
    font_scale = max(0.4, min(0.85, image.shape[1] / 1700))
    cv2.putText(
        image,
        title,
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        legend,
        (12, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale * 0.82,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"failed to save visualization: {output_path}")


def _save_plots(error_matrix: np.ndarray, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    values = error_matrix.reshape(-1)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(values, bins=35, color="#4472c4", edgecolor="white")
    axis.axvline(values.mean(), color="#c00000", linestyle="--", label=f"mean {values.mean():.1f}px")
    axis.axvline(
        np.median(values),
        color="#70ad47",
        linestyle="--",
        label=f"median {np.median(values):.1f}px",
    )
    axis.set(xlabel="Euclidean keypoint error (pixels)", ylabel="count", title="Point error distribution")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "error_distribution.png", dpi=180)
    plt.close(figure)

    means = error_matrix.mean(axis=0)
    p95s = np.percentile(error_matrix, 95, axis=0)
    figure, axis = plt.subplots(figsize=(12, 5))
    positions = np.arange(len(KEYPOINT_NAMES))
    axis.bar(positions, means, color="#4472c4", label="mean")
    axis.scatter(positions, p95s, color="#c00000", s=25, label="p95", zorder=3)
    axis.set_xticks(positions, KEYPOINT_NAMES, rotation=55, ha="right")
    axis.set(ylabel="error (pixels)", title="Per-keypoint error")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "per_keypoint_error.png", dpi=180)
    plt.close(figure)


def _make_contact_sheet(paths: list[Path], output_path: Path) -> None:
    if not paths:
        return
    columns = 4
    tile_width, tile_height = 420, 520
    rows = math.ceil(len(paths) / columns)
    canvas = np.full((rows * tile_height, columns * tile_width, 3), 28, dtype=np.uint8)
    for slot, path in enumerate(paths):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        scale = min(tile_width / image.shape[1], tile_height / image.shape[0])
        width = max(1, round(image.shape[1] * scale))
        height = max(1, round(image.shape[0] * scale))
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        x0 = (slot % columns) * tile_width + (tile_width - width) // 2
        y0 = (slot // columns) * tile_height + (tile_height - height) // 2
        canvas[y0 : y0 + height, x0 : x0 + width] = resized
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"failed to save contact sheet: {output_path}")


def save_evaluation(
    cases: list[CasePrediction],
    names: Any,
    dataset_root: Path,
    output_dir: Path,
    *,
    model_metadata: dict[str, Any] | None = None,
    save_all: bool = True,
    save_worst: int = 20,
) -> dict[str, Any]:
    """Compute unified original-image metrics and save all evaluation artifacts."""
    if not cases:
        raise RuntimeError("no evaluation cases were provided")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(dataset_root)
    records: list[dict[str, Any]] = []
    errors_all: list[np.ndarray] = []
    normalized_all: list[np.ndarray] = []
    group_errors: dict[str, list[np.ndarray]] = defaultdict(list)
    group_normalized: dict[str, list[np.ndarray]] = defaultdict(list)
    detected = 0
    class_correct = 0

    for case in cases:
        image = cv2.imread(str(case.image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"failed to read image: {case.image_path}")
        height, width = image.shape
        diagonal = math.hypot(width, height)
        if case.gt_points.shape != (len(KEYPOINT_NAMES), 2):
            raise RuntimeError(f"unexpected GT shape for {case.image_path}: {case.gt_points.shape}")
        if case.pred_points is None:
            errors = np.full(len(KEYPOINT_NAMES), diagonal, dtype=np.float64)
            predicted = None
        else:
            predicted = np.asarray(case.pred_points, dtype=np.float64)
            if predicted.shape != (len(KEYPOINT_NAMES), 2) or not np.isfinite(predicted).all():
                raise RuntimeError(f"invalid prediction for {case.image_path}: {predicted.shape}")
            errors = np.linalg.norm(predicted - case.gt_points, axis=1)
            detected += 1
        normalized = errors / diagonal
        class_correct += int(case.pred_class is not None and case.pred_class == case.gt_class)
        errors_all.append(errors)
        normalized_all.append(normalized)
        record_manifest = manifest.get(case.image_path.name, {})
        source = str(record_manifest.get("source", "unknown"))
        gt_name = class_name(names, case.gt_class)
        group_errors[f"class:{gt_name}"].append(errors)
        group_normalized[f"class:{gt_name}"].append(normalized)
        group_errors[f"source:{source}"].append(errors)
        group_normalized[f"source:{source}"].append(normalized)
        records.append(
            {
                "image_path": case.image_path,
                "image": case.image_path.name,
                "source": source,
                "gt_class": case.gt_class,
                "gt_name": gt_name,
                "pred_class": case.pred_class,
                "pred_name": class_name(names, case.pred_class),
                "class_correct": int(case.pred_class == case.gt_class) if case.pred_class is not None else 0,
                "detected": int(predicted is not None),
                "confidence": case.confidence,
                "gt_points": case.gt_points,
                "pred_points": predicted,
                "errors": errors,
                "normalized": normalized,
                "mean_error_px": float(errors.mean()),
                "median_error_px": float(np.median(errors)),
                "max_error_px": float(errors.max()),
                "mean_error_image_diag_pct": float(normalized.mean() * 100),
                "pck_0.5pct_image_diag": float((normalized <= 0.005).mean()),
                "pck_1pct_image_diag": float((normalized <= 0.01).mean()),
                "pck_2pct_image_diag": float((normalized <= 0.02).mean()),
            }
        )

    error_matrix = np.stack(errors_all)
    normalized_matrix = np.stack(normalized_all)
    records.sort(key=lambda item: item["mean_error_image_diag_pct"], reverse=True)
    case_fields = (
        "rank",
        "image",
        "source",
        "gt_name",
        "pred_name",
        "class_correct",
        "detected",
        "confidence",
        "mean_error_px",
        "median_error_px",
        "max_error_px",
        "mean_error_image_diag_pct",
        "pck_0.5pct_image_diag",
        "pck_1pct_image_diag",
        "pck_2pct_image_diag",
    )
    with (output_dir / "cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=case_fields)
        writer.writeheader()
        for rank, record in enumerate(records, start=1):
            writer.writerow({"rank": rank, **{key: record[key] for key in case_fields if key != "rank"}})

    point_fields = (
        "index",
        "keypoint",
        "mean_error_px",
        "median_error_px",
        "p95_error_px",
        "max_error_px",
        "mean_error_image_diag_pct",
        "pck_0.5pct_image_diag",
        "pck_1pct_image_diag",
        "pck_2pct_image_diag",
    )
    with (output_dir / "per_keypoint.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=point_fields)
        writer.writeheader()
        for index, name in enumerate(KEYPOINT_NAMES):
            block = _metric_block(error_matrix[:, index], normalized_matrix[:, index])
            writer.writerow({"index": index + 1, "keypoint": name, **block})

    corner_pairs = normalized_matrix[:, 7:].reshape(-1, 6, 2)
    summary: dict[str, Any] = {
        "images": len(records),
        "keypoints_per_image": len(KEYPOINT_NAMES),
        "missing_prediction_penalty": "one original-image diagonal per point",
        "point_metrics": _metric_block(error_matrix.reshape(-1), normalized_matrix.reshape(-1)),
        "center_metrics": _metric_block(
            error_matrix[:, :7].reshape(-1),
            normalized_matrix[:, :7].reshape(-1),
        ),
        "corner_metrics": _metric_block(
            error_matrix[:, 7:].reshape(-1),
            normalized_matrix[:, 7:].reshape(-1),
        ),
        "vu_pair_pck": {
            "pck_0.5pct_image_diag": float(np.all(corner_pairs <= 0.005, axis=-1).mean()),
            "pck_1pct_image_diag": float(np.all(corner_pairs <= 0.01, axis=-1).mean()),
            "pck_2pct_image_diag": float(np.all(corner_pairs <= 0.02, axis=-1).mean()),
        },
        "detection_coverage": detected / len(records),
        "class_accuracy": class_correct / len(records),
        "groups": {},
        "model": model_metadata or {},
    }
    for group in sorted(group_errors):
        group_error = np.stack(group_errors[group]).reshape(-1)
        group_norm = np.stack(group_normalized[group]).reshape(-1)
        summary["groups"][group] = _metric_block(group_error, group_norm)
    (output_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    all_dir = output_dir / "all_cases"
    worst_dir = output_dir / "worst_cases"
    worst_paths: list[Path] = []
    for rank, record in enumerate(records, start=1):
        if save_all:
            _draw_case(record, all_dir / f"{rank:04d}_{record['image_path'].stem}.jpg")
        if rank <= max(0, save_worst):
            path = worst_dir / f"{rank:03d}_{record['image_path'].stem}.jpg"
            _draw_case(record, path)
            worst_paths.append(path)
    _make_contact_sheet(worst_paths, output_dir / "worst_cases_contact_sheet.jpg")
    _save_plots(error_matrix, output_dir)
    return summary
