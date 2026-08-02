#!/usr/bin/env python3
"""Shared EfficientNet-B0 model, data pipeline, losses, and metrics for Stage 1."""

from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TORCH_HOME = Path(os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".cache" / "torch")))
torch.hub.set_dir(str(TORCH_HOME / "hub"))


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
NUM_KEYPOINTS = len(KEYPOINT_NAMES)
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_image_size(size: int | Sequence[int]) -> tuple[int, int]:
    """Return a fixed (height, width), accepting one square size or two dimensions."""
    if isinstance(size, int):
        values = [size]
    else:
        values = [int(value) for value in size]
    if len(values) == 1:
        height = width = values[0]
    elif len(values) == 2:
        height, width = values
    else:
        raise ValueError("image size must contain one value (square) or two values (height width)")
    if height < 1 or width < 1:
        raise ValueError("image height and width must be positive")
    return height, width


def resolve_dataset_split(data_yaml: Path, split: str) -> tuple[dict[str, Any], list[Path], Path]:
    """Resolve one directory-based split from the existing YOLO dataset YAML."""
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(config.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    split_value = config.get(split)
    if not isinstance(split_value, str):
        raise RuntimeError(f"{split!r} must be a directory path in {data_yaml}")
    image_dir = Path(split_value)
    if not image_dir.is_absolute():
        image_dir = root / image_dir
    label_dir = root / "labels" / image_dir.name
    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.casefold() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    )
    if not image_paths or not label_dir.is_dir():
        raise RuntimeError(f"missing images or labels for split {split!r}")
    return config, image_paths, label_dir


def read_pose_label(label_path: Path, width: int, height: int) -> tuple[int, np.ndarray]:
    """Read the one-instance YOLO pose label and return original-pixel keypoints."""
    lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"expected one pose instance in {label_path}, found {len(lines)}")
    values = np.asarray([float(value) for value in lines[0].split()], dtype=np.float32)
    expected = 5 + NUM_KEYPOINTS * 3
    if len(values) != expected:
        raise RuntimeError(f"expected {expected} fields in {label_path}, found {len(values)}")
    class_id = int(values[0])
    keypoints = values[5:].reshape(NUM_KEYPOINTS, 3)
    if not np.all(keypoints[:, 2] > 0):
        raise RuntimeError(f"all keypoints must be visible in {label_path}")
    points = keypoints[:, :2].copy()
    points[:, 0] *= width
    points[:, 1] *= height
    return class_id, points


def letterbox(
    image: np.ndarray,
    points: np.ndarray,
    size: int | Sequence[int],
) -> tuple[np.ndarray, np.ndarray, float, tuple[int, int]]:
    """Resize without aspect-ratio distortion and transform keypoints."""
    height, width = image.shape[:2]
    input_height, input_width = normalize_image_size(size)
    scale = min(input_width / width, input_height / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    pad_x = (input_width - resized_width) // 2
    pad_y = (input_height - resized_height) // 2
    canvas = np.zeros((input_height, input_width, 3), dtype=np.uint8)
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    transformed = points.astype(np.float32).copy()
    transformed[:, 0] = transformed[:, 0] * scale + pad_x
    transformed[:, 1] = transformed[:, 1] * scale + pad_y
    return canvas, transformed, scale, (pad_x, pad_y)


def random_affine(
    image: np.ndarray,
    points: np.ndarray,
    degrees: float,
    translate: float,
    scale_gain: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply an affine transform while keeping every supervised point inside the image."""
    height, width = image.shape[:2]
    center = ((width - 1) / 2, (height - 1) / 2)
    for _ in range(12):
        angle = float(np.random.uniform(-degrees, degrees))
        scale = float(np.random.uniform(1.0 - scale_gain, 1.0 + scale_gain))
        tx = float(np.random.uniform(-translate, translate) * width)
        ty = float(np.random.uniform(-translate, translate) * height)
        matrix = cv2.getRotationMatrix2D(center, angle, scale).astype(np.float32)
        matrix[:, 2] += (tx, ty)
        transformed = np.concatenate(
            [points, np.ones((len(points), 1), dtype=np.float32)], axis=1
        ) @ matrix.T
        inside = (
            (transformed[:, 0] >= 2)
            & (transformed[:, 0] <= width - 3)
            & (transformed[:, 1] >= 2)
            & (transformed[:, 1] <= height - 3)
        )
        if np.all(inside):
            warped = cv2.warpAffine(
                image,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            return warped, transformed
    return image, points


def intensity_augment(image: np.ndarray) -> np.ndarray:
    """Apply conservative radiograph intensity augmentation."""
    array = image.astype(np.float32) / 255.0
    padding_mask = np.all(image == 0, axis=2)
    gamma = float(np.random.uniform(0.90, 1.10))
    contrast = float(np.random.uniform(0.90, 1.10))
    brightness = float(np.random.uniform(-0.03, 0.03))
    array = np.power(np.clip(array, 0.0, 1.0), gamma)
    array = (array - 0.5) * contrast + 0.5 + brightness
    if np.random.random() < 0.25:
        array += np.random.normal(0.0, 0.008, array.shape).astype(np.float32)
    array = np.clip(array, 0.0, 1.0)
    array[padding_mask] = 0.0
    return array


class SpineKeypointDataset(Dataset):
    """One radiograph, one class and one fixed ordered set of 19 keypoints."""

    def __init__(
        self,
        data_yaml: Path,
        split: str,
        imgsz: int | Sequence[int],
        augment: bool = False,
        degrees: float = 5.0,
        translate: float = 0.03,
        scale: float = 0.10,
    ) -> None:
        self.config, self.image_paths, self.label_dir = resolve_dataset_split(data_yaml, split)
        self.image_size = normalize_image_size(imgsz)
        self.augment = augment
        self.degrees = degrees
        self.translate = translate
        self.scale_gain = scale

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path = self.image_paths[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read image: {image_path}")
        height, width = image.shape[:2]
        class_id, original_points = read_pose_label(
            self.label_dir / f"{image_path.stem}.txt", width, height
        )
        image, input_points, resize_scale, pad = letterbox(
            image, original_points, self.image_size
        )
        if self.augment:
            image, input_points = random_affine(
                image, input_points, self.degrees, self.translate, self.scale_gain
            )
            array = intensity_augment(image)
        else:
            array = image.astype(np.float32) / 255.0
        # torchvision ImageNet weights expect RGB channel order.
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        array = (array - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(array.transpose(2, 0, 1).copy()).float()
        input_height, input_width = self.image_size
        divisor = np.asarray([input_width - 1, input_height - 1], dtype=np.float32)
        target = torch.from_numpy((input_points / divisor).astype(np.float32))
        return {
            "image": tensor,
            "target": target.clamp(0.0, 1.0),
            "class_id": torch.tensor(class_id, dtype=torch.long),
            "path": str(image_path),
            "original_points": torch.from_numpy(original_points.astype(np.float32)),
            "original_shape": torch.tensor([height, width], dtype=torch.float32),
            "resize_scale": torch.tensor(resize_scale, dtype=torch.float32),
            "pad": torch.tensor(pad, dtype=torch.float32),
        }


def conv_norm_act(in_channels: int, out_channels: int, kernel_size: int = 1) -> nn.Sequential:
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
        nn.GroupNorm(8, out_channels),
        nn.SiLU(inplace=True),
    )


class EfficientNetKeypointModel(nn.Module):
    """EfficientNet-B0 backbone with a lightweight stride-4 FPN heatmap head."""

    def __init__(self, num_keypoints: int = NUM_KEYPOINTS, num_classes: int = 2, pretrained: bool = True) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        base = efficientnet_b0(weights=weights)
        self.backbone = base.features
        fpn_channels = 64
        self.lat4 = conv_norm_act(24, fpn_channels)
        self.lat8 = conv_norm_act(40, fpn_channels)
        self.lat16 = conv_norm_act(112, fpn_channels)
        self.lat32 = conv_norm_act(1280, fpn_channels)
        self.smooth16 = conv_norm_act(fpn_channels, fpn_channels, 3)
        self.smooth8 = conv_norm_act(fpn_channels, fpn_channels, 3)
        self.smooth4 = conv_norm_act(fpn_channels, fpn_channels, 3)
        self.heatmap_head = nn.Sequential(
            conv_norm_act(fpn_channels, fpn_channels, 3),
            nn.Conv2d(fpn_channels, num_keypoints, 1),
        )
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features: dict[int, torch.Tensor] = {}
        x = image
        for index, stage in enumerate(self.backbone):
            x = stage(x)
            if index in {2, 3, 5, 8}:
                features[index] = x
        p32 = self.lat32(features[8])
        p16 = self.smooth16(
            self.lat16(features[5]) + F.interpolate(p32, size=features[5].shape[-2:], mode="bilinear", align_corners=False)
        )
        p8 = self.smooth8(
            self.lat8(features[3]) + F.interpolate(p16, size=features[3].shape[-2:], mode="bilinear", align_corners=False)
        )
        p4 = self.smooth4(
            self.lat4(features[2]) + F.interpolate(p8, size=features[2].shape[-2:], mode="bilinear", align_corners=False)
        )
        heatmaps = self.heatmap_head(p4)
        coordinates = soft_argmax_2d(heatmaps)
        pooled = F.adaptive_avg_pool2d(features[8], 1).flatten(1)
        class_logits = self.classifier(pooled)
        return {"heatmaps": heatmaps, "coordinates": coordinates, "class_logits": class_logits}


def soft_argmax_2d(heatmaps: torch.Tensor) -> torch.Tensor:
    """Return differentiable normalized (x, y) coordinates from K spatial logits."""
    batch, keypoints, height, width = heatmaps.shape
    probability = heatmaps.reshape(batch, keypoints, -1).softmax(dim=-1)
    xs = torch.linspace(0.0, 1.0, width, device=heatmaps.device, dtype=heatmaps.dtype)
    ys = torch.linspace(0.0, 1.0, height, device=heatmaps.device, dtype=heatmaps.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    x = (probability * grid_x.reshape(1, 1, -1)).sum(dim=-1)
    y = (probability * grid_y.reshape(1, 1, -1)).sum(dim=-1)
    return torch.stack([x, y], dim=-1)


def gaussian_targets(
    coordinates: torch.Tensor, height: int, width: int, sigma: float = 2.0
) -> torch.Tensor:
    """Create normalized Gaussian probability maps around normalized target coordinates."""
    center_x = coordinates[..., 0] * (width - 1)
    center_y = coordinates[..., 1] * (height - 1)
    xs = torch.arange(width, device=coordinates.device, dtype=coordinates.dtype)
    ys = torch.arange(height, device=coordinates.device, dtype=coordinates.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    squared_distance = (
        (grid_x.reshape(1, 1, height, width) - center_x[..., None, None]).square()
        + (grid_y.reshape(1, 1, height, width) - center_y[..., None, None]).square()
    )
    target = torch.exp(-squared_distance / (2 * sigma**2))
    return target / target.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)


def keypoint_loss(
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    class_id: torch.Tensor,
    sigma: float = 2.0,
    coordinate_gain: float = 10.0,
    structure_gain: float = 2.0,
    class_gain: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Heatmap, coordinate, spine-structure and auxiliary class supervision."""
    heatmaps = outputs["heatmaps"]
    target_maps = gaussian_targets(target, heatmaps.shape[-2], heatmaps.shape[-1], sigma)
    log_probability = F.log_softmax(heatmaps.flatten(2), dim=-1)
    heatmap_loss = F.kl_div(
        log_probability,
        target_maps.flatten(2),
        reduction="none",
    ).sum(dim=-1).mean()
    coordinate_loss = F.smooth_l1_loss(
        outputs["coordinates"], target, beta=0.01, reduction="mean"
    )
    structure_loss, structure_parts = spine_structure_loss(outputs["coordinates"], target)
    classification_loss = F.cross_entropy(outputs["class_logits"], class_id)
    total = (
        heatmap_loss
        + coordinate_gain * coordinate_loss
        + structure_gain * structure_loss
        + class_gain * classification_loss
    )
    return total, {
        "total_loss": float(total.detach()),
        "heatmap_loss": float(heatmap_loss.detach()),
        "coordinate_loss": float(coordinate_loss.detach()),
        "structure_loss": float(structure_loss.detach()),
        "adjacent_loss": float(structure_parts["adjacent_loss"].detach()),
        "pair_loss": float(structure_parts["pair_loss"].detach()),
        "order_loss": float(structure_parts["order_loss"].detach()),
        "class_loss": float(classification_loss.detach()),
    }


def spine_structure_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    beta: float = 0.01,
    order_margin: float = 0.002,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Preserve target-specific chains, point pairs and superior-inferior order.

    The loss learns relative vectors from each individual target rather than enforcing
    a population-average spine, so unusual but correctly annotated anatomy remains valid.
    """
    if predicted.shape != target.shape or predicted.shape[-2:] != (NUM_KEYPOINTS, 2):
        raise ValueError(
            f"expected matching [batch, {NUM_KEYPOINTS}, 2] tensors, got "
            f"{tuple(predicted.shape)} and {tuple(target.shape)}"
        )

    # Three ordered superior-to-inferior chains: B1..B7, V1P1..V6P1 and V1P2..V6P2.
    chain_indices = (
        torch.tensor([0, 1, 2, 3, 4, 5, 6], device=predicted.device),
        torch.tensor([7, 9, 11, 13, 15, 17], device=predicted.device),
        torch.tensor([8, 10, 12, 14, 16, 18], device=predicted.device),
    )
    predicted_vectors: list[torch.Tensor] = []
    target_vectors: list[torch.Tensor] = []
    order_penalties: list[torch.Tensor] = []
    for indices in chain_indices:
        predicted_chain = predicted.index_select(1, indices)
        target_chain = target.index_select(1, indices)
        predicted_delta = predicted_chain[:, 1:] - predicted_chain[:, :-1]
        target_delta = target_chain[:, 1:] - target_chain[:, :-1]
        predicted_vectors.append(predicted_delta)
        target_vectors.append(target_delta)

        # Respect the annotated vertical direction while allowing arbitrary spacing.
        target_direction = torch.where(
            target_delta[..., 1] >= 0,
            torch.ones_like(target_delta[..., 1]),
            -torch.ones_like(target_delta[..., 1]),
        )
        directed_step = predicted_delta[..., 1] * target_direction
        order_penalties.append(F.relu(order_margin - directed_step))

    adjacent_loss = F.smooth_l1_loss(
        torch.cat(predicted_vectors, dim=1),
        torch.cat(target_vectors, dim=1),
        beta=beta,
        reduction="mean",
    )

    # Six VxP1/VxP2 vectors preserve each annotated within-level pair relationship.
    p1_indices = torch.tensor([7, 9, 11, 13, 15, 17], device=predicted.device)
    p2_indices = torch.tensor([8, 10, 12, 14, 16, 18], device=predicted.device)
    predicted_pairs = predicted.index_select(1, p2_indices) - predicted.index_select(1, p1_indices)
    target_pairs = target.index_select(1, p2_indices) - target.index_select(1, p1_indices)
    pair_loss = F.smooth_l1_loss(
        predicted_pairs, target_pairs, beta=beta, reduction="mean"
    )
    order_loss = torch.cat(order_penalties, dim=1).mean()
    structure_loss = 0.5 * adjacent_loss + 0.3 * pair_loss + 0.2 * order_loss
    return structure_loss, {
        "adjacent_loss": adjacent_loss,
        "pair_loss": pair_loss,
        "order_loss": order_loss,
    }


def decode_original_coordinates(
    coordinates: torch.Tensor,
    imgsz: int | Sequence[int],
    resize_scale: torch.Tensor,
    pad: torch.Tensor,
    original_shape: torch.Tensor,
) -> torch.Tensor:
    """Map normalized letterbox coordinates back to original image pixels."""
    input_height, input_width = normalize_image_size(imgsz)
    scale = torch.tensor(
        [input_width - 1, input_height - 1],
        dtype=torch.float32,
        device=coordinates.device,
    )
    points = coordinates.float().clone() * scale
    points[..., 0] = (points[..., 0] - pad[:, None, 0]) / resize_scale[:, None]
    points[..., 1] = (points[..., 1] - pad[:, None, 1]) / resize_scale[:, None]
    widths = original_shape[:, 1]
    heights = original_shape[:, 0]
    points[..., 0] = torch.maximum(
        torch.zeros_like(points[..., 0]), torch.minimum(points[..., 0], widths[:, None] - 1)
    )
    points[..., 1] = torch.maximum(
        torch.zeros_like(points[..., 1]), torch.minimum(points[..., 1], heights[:, None] - 1)
    )
    return points


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    imgsz: int | Sequence[int],
    amp: bool = True,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Evaluate only point geometry and the auxiliary C/L classification."""
    model.eval()
    errors_px: list[float] = []
    normalized_errors: list[float] = []
    class_correct = 0
    cases: list[dict[str, Any]] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            outputs = model(images)
        predicted = decode_original_coordinates(
            outputs["coordinates"].cpu(),
            imgsz,
            batch["resize_scale"],
            batch["pad"],
            batch["original_shape"],
        )
        target = batch["original_points"].float()
        distances = torch.linalg.vector_norm(predicted - target, dim=-1)
        diagonals = torch.linalg.vector_norm(batch["original_shape"].float(), dim=-1)
        normalized = distances / diagonals[:, None]
        predicted_class = outputs["class_logits"].argmax(dim=1).cpu()
        class_correct += int((predicted_class == batch["class_id"]).sum())
        errors_px.extend(distances.flatten().tolist())
        normalized_errors.extend(normalized.flatten().tolist())
        for index, path in enumerate(batch["path"]):
            cases.append(
                {
                    "path": path,
                    "gt_class": int(batch["class_id"][index]),
                    "pred_class": int(predicted_class[index]),
                    "pred_points": predicted[index].numpy(),
                    "gt_points": target[index].numpy(),
                    "mean_error_px": float(distances[index].mean()),
                    "max_error_px": float(distances[index].max()),
                    "mean_error_image_diag": float(normalized[index].mean()),
                }
            )
    pixel_array = np.asarray(errors_px, dtype=np.float64)
    norm_array = np.asarray(normalized_errors, dtype=np.float64)
    metrics = {
        "mean_error_px": float(pixel_array.mean()),
        "median_error_px": float(np.median(pixel_array)),
        "p95_error_px": float(np.percentile(pixel_array, 95)),
        "mean_error_image_diag_pct": float(norm_array.mean() * 100),
        "pck_0.5pct_image_diag": float((norm_array <= 0.005).mean()),
        "pck_1pct_image_diag": float((norm_array <= 0.01).mean()),
        "pck_2pct_image_diag": float((norm_array <= 0.02).mean()),
        "class_accuracy": class_correct / len(loader.dataset),
    }
    return metrics, cases


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python RNG independently in every dataloader worker."""
    del worker_id
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)
