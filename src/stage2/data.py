"""Scored Zhongri VU samples, canonical crops, and conservative online augmentation."""

from __future__ import annotations

import json
import math
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler, get_worker_info


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ZHONGRI_ROOT = PROJECT_ROOT / "raw_data/zhongri/1-after-trim"
DEFAULT_CROP_SIZE = 256
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

CERVICAL_LEVELS = ("C2-C3", "C3-C4", "C4-C5", "C5-C6", "C6-C7", "C7-T1")
LUMBAR_LEVELS = ("T12-L1", "L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1")


@dataclass(frozen=True)
class VUSample:
    """One VU with two independent mSASSS corner labels."""

    sample_id: str
    patient_id: str
    view: str
    level_index: int
    level_name: str
    image_id: int
    annotation_id: int
    image_path: Path
    image_width: int
    image_height: int
    upper_center: tuple[float, float]
    lower_center: tuple[float, float]
    up_corner: tuple[float, float]
    down_corner: tuple[float, float]
    up_score: int
    down_score: int


@dataclass(frozen=True)
class CropJitter:
    """Perturbation of the Stage-1-derived oriented crop frame."""

    rotation_deg: float = 0.0
    translate_x_fraction: float = 0.0
    translate_y_fraction: float = 0.0
    field_scale: float = 1.0

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [self.rotation_deg, self.translate_x_fraction, self.translate_y_fraction, self.field_scale],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class AugmentationConfig:
    """Safe defaults for fine-grained lateral-spine scoring."""

    rotation_deg: float = 3.0
    translation_fraction: float = 0.04
    field_scale: float = 0.10
    gamma: float = 0.10
    contrast: float = 0.10
    brightness: float = 0.03
    noise_probability: float = 0.25
    noise_sigma: float = 0.008
    blur_probability: float = 0.15
    blur_sigma_max: float = 0.8

    def validate(self) -> None:
        magnitudes = (
            self.rotation_deg,
            self.translation_fraction,
            self.field_scale,
            self.gamma,
            self.contrast,
            self.brightness,
            self.noise_probability,
            self.noise_sigma,
            self.blur_probability,
            self.blur_sigma_max,
        )
        if min(magnitudes) < 0:
            raise ValueError("augmentation magnitudes cannot be negative")
        if self.field_scale >= 1:
            raise ValueError("field_scale must be below 1")
        if not 0 <= self.noise_probability <= 1 or not 0 <= self.blur_probability <= 1:
            raise ValueError("augmentation probabilities must be in [0, 1]")


def decode_image(path: Path) -> np.ndarray:
    """Read non-ASCII paths and always return three-channel BGR."""
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"cannot decode image: {path}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def encode_image(path: Path, image: np.ndarray, quality: int = 95) -> None:
    """Write an image without relying on OpenCV path encoding."""
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if path.suffix.casefold() in {".jpg", ".jpeg"} else []
    ok, encoded = cv2.imencode(path.suffix.casefold(), image, params)
    if not ok:
        raise RuntimeError(f"cannot encode image: {path}")
    encoded.tofile(path)


def _visible_points(annotation: dict[str, Any], expected: int) -> list[tuple[float, float]]:
    values = annotation.get("keypoints")
    if not isinstance(values, list) or len(values) != expected * 3:
        raise RuntimeError(f"annotation {annotation.get('id')} must contain {expected} keypoints")
    points: list[tuple[float, float]] = []
    for index in range(0, len(values), 3):
        x, y, visibility = values[index : index + 3]
        if int(visibility) <= 0:
            raise RuntimeError(f"invisible point in annotation {annotation.get('id')}")
        points.append((float(x), float(y)))
    return points


def _patient_and_view(file_name: str) -> tuple[str, str]:
    stem = Path(file_name).stem
    if "-" not in stem:
        raise RuntimeError(f"cannot parse patient/view from {file_name}")
    patient, view = stem.rsplit("-", 1)
    if view not in {"C", "L"}:
        raise RuntimeError(f"unsupported view {view!r} in {file_name}")
    return patient, view


def _ordered_vus(
    centers: Sequence[tuple[float, float]], annotations: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    interval_midpoints = [
        ((centers[i][0] + centers[i + 1][0]) / 2, (centers[i][1] + centers[i + 1][1]) / 2)
        for i in range(6)
    ]
    assigned: dict[int, dict[str, Any]] = {}
    for annotation in annotations:
        corners = _visible_points(annotation, 2)
        midpoint = ((corners[0][0] + corners[1][0]) / 2, (corners[0][1] + corners[1][1]) / 2)
        interval = min(
            range(6),
            key=lambda i: (midpoint[0] - interval_midpoints[i][0]) ** 2
            + (midpoint[1] - interval_midpoints[i][1]) ** 2,
        )
        if interval in assigned:
            raise RuntimeError(f"duplicate VU assignment at interval {interval + 1}")
        assigned[interval] = annotation
    if set(assigned) != set(range(6)):
        raise RuntimeError(f"incomplete VU assignments: {sorted(assigned)}")
    return [assigned[index] for index in range(6)]


def load_zhongri_samples(root: Path = DEFAULT_ZHONGRI_ROOT) -> list[VUSample]:
    """Read all 696 scored VUs; unscored external data are never included."""
    root = root.expanduser().resolve()
    annotation_path = root / "annotations/person_keypoints_default.json"
    image_dir = root / "images/default"
    if not annotation_path.is_file() or not image_dir.is_dir():
        raise FileNotFoundError(f"invalid Zhongri root: {root}")
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    category_names = {int(item["id"]): str(item["name"]) for item in coco["categories"]}
    if set(category_names.values()) != {"bone", "vu"}:
        raise RuntimeError(f"unexpected categories: {category_names}")
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        by_image[int(annotation["image_id"])].append(annotation)

    samples: list[VUSample] = []
    for image_info in sorted(coco["images"], key=lambda item: str(item["file_name"])):
        image_id = int(image_info["id"])
        file_name = str(image_info["file_name"])
        patient_id, view = _patient_and_view(file_name)
        image_path = image_dir / file_name
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        annotations = by_image[image_id]
        bones = [item for item in annotations if category_names[int(item["category_id"])] == "bone"]
        vus = [item for item in annotations if category_names[int(item["category_id"])] == "vu"]
        if len(bones) != 1 or len(vus) != 6:
            raise RuntimeError(f"{file_name}: expected one bone and six VUs")
        centers = _visible_points(bones[0], 7)
        ordered = _ordered_vus(centers, vus)
        levels = CERVICAL_LEVELS if view == "C" else LUMBAR_LEVELS
        for level_index, annotation in enumerate(ordered, start=1):
            attributes = annotation.get("attributes", {})
            if "up" not in attributes or "down" not in attributes:
                raise RuntimeError(f"scores missing in annotation {annotation.get('id')}")
            up_score, down_score = int(attributes["up"]), int(attributes["down"])
            if up_score not in range(4) or down_score not in range(4):
                raise RuntimeError(f"score outside 0..3 in annotation {annotation.get('id')}")
            corners = _visible_points(annotation, 2)
            samples.append(
                VUSample(
                    sample_id=f"{Path(file_name).stem}__V{level_index}",
                    patient_id=patient_id,
                    view=view,
                    level_index=level_index,
                    level_name=levels[level_index - 1],
                    image_id=image_id,
                    annotation_id=int(annotation["id"]),
                    image_path=image_path,
                    image_width=int(image_info["width"]),
                    image_height=int(image_info["height"]),
                    upper_center=centers[level_index - 1],
                    lower_center=centers[level_index],
                    up_corner=corners[0],
                    down_corner=corners[1],
                    up_score=up_score,
                    down_score=down_score,
                )
            )
    if len(samples) != 696:
        raise RuntimeError(f"expected 696 scored Zhongri VUs, found {len(samples)}")
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise RuntimeError("duplicate Stage-2 sample IDs")
    return samples


def sample_crop_jitter(rng: np.random.Generator, config: AugmentationConfig) -> CropJitter:
    """Draw one crop perturbation; mirroring is intentionally absent."""
    config.validate()
    return CropJitter(
        rotation_deg=float(rng.uniform(-config.rotation_deg, config.rotation_deg)),
        translate_x_fraction=float(rng.uniform(-config.translation_fraction, config.translation_fraction)),
        translate_y_fraction=float(rng.uniform(-config.translation_fraction, config.translation_fraction)),
        field_scale=float(rng.uniform(1.0 - config.field_scale, 1.0 + config.field_scale)),
    )


def crop_quad(sample: VUSample, jitter: CropJitter | None = None) -> np.ndarray:
    """Return TL, TR, BR, BL source coordinates for an oriented square VU crop."""
    jitter = jitter or CropJitter()
    upper = np.asarray(sample.upper_center, dtype=np.float64)
    lower = np.asarray(sample.lower_center, dtype=np.float64)
    direction = lower - upper
    side = float(np.linalg.norm(direction))
    if side < 2:
        raise RuntimeError(f"degenerate center interval in {sample.sample_id}")
    unit_y = direction / side
    normal = np.asarray([-unit_y[1], unit_y[0]], dtype=np.float64)
    center_midpoint = (upper + lower) / 2
    corner_midpoint = (
        np.asarray(sample.up_corner, dtype=np.float64) + np.asarray(sample.down_corner, dtype=np.float64)
    ) / 2
    if float(np.dot(corner_midpoint - center_midpoint, normal)) < 0:
        normal = -normal
    unit_x = -normal  # output x increases from anterior to posterior
    crop_center = center_midpoint + normal * side / 2
    crop_center += unit_x * jitter.translate_x_fraction * side
    crop_center += unit_y * jitter.translate_y_fraction * side
    angle = math.radians(jitter.rotation_deg)
    rotation = np.asarray([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    unit_x, unit_y = rotation @ unit_x, rotation @ unit_y
    half_x = unit_x * side * jitter.field_scale / 2
    half_y = unit_y * side * jitter.field_scale / 2
    return np.asarray(
        [
            crop_center - half_x - half_y,
            crop_center + half_x - half_y,
            crop_center + half_x + half_y,
            crop_center - half_x + half_y,
        ],
        dtype=np.float32,
    )


def extract_vu_crop(
    image: np.ndarray,
    sample: VUSample,
    output_size: int = DEFAULT_CROP_SIZE,
    jitter: CropJitter | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Perspective-warp a VU to a canonical square and return its source quad."""
    if output_size < 32:
        raise ValueError("output_size must be at least 32")
    source = crop_quad(sample, jitter)
    matrix = crop_perspective_transform(source, output_size)
    crop = cv2.warpPerspective(
        image,
        matrix,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return crop, source


def crop_perspective_transform(source_quad: np.ndarray, output_size: int) -> np.ndarray:
    """Return the source-image to canonical-crop homography."""
    source = np.asarray(source_quad, dtype=np.float32)
    if source.shape != (4, 2):
        raise ValueError(f"source_quad must have shape [4,2], got {source.shape}")
    if output_size < 32:
        raise ValueError("output_size must be at least 32")
    maximum = float(output_size - 1)
    target = np.asarray(
        [[0, 0], [maximum, 0], [maximum, maximum], [0, maximum]],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(source, target)


def transform_points_to_crop(
    points: Sequence[tuple[float, float]] | np.ndarray,
    source_quad: np.ndarray,
    output_size: int = DEFAULT_CROP_SIZE,
) -> np.ndarray:
    """Map original-image points into the same canonical crop as the image."""
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"points must have shape [N,2], got {array.shape}")
    matrix = crop_perspective_transform(source_quad, output_size)
    transformed = cv2.perspectiveTransform(array[:, None, :], matrix)[:, 0, :]
    if not np.isfinite(transformed).all():
        raise RuntimeError("non-finite point produced by VU crop transform")
    return transformed.astype(np.float32, copy=False)


def augment_intensity(
    image: np.ndarray, rng: np.random.Generator, config: AugmentationConfig
) -> np.ndarray:
    """Apply mild grayscale-preserving X-ray intensity augmentation."""
    config.validate()
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    padding_mask = grayscale == 0
    array = grayscale.astype(np.float32) / 255.0
    gamma = float(rng.uniform(1.0 - config.gamma, 1.0 + config.gamma))
    contrast = float(rng.uniform(1.0 - config.contrast, 1.0 + config.contrast))
    brightness = float(rng.uniform(-config.brightness, config.brightness))
    array = np.power(np.clip(array, 0.0, 1.0), gamma)
    array = (array - 0.5) * contrast + 0.5 + brightness
    if rng.random() < config.noise_probability and config.noise_sigma:
        array += rng.normal(0.0, config.noise_sigma, array.shape).astype(np.float32)
    array = np.clip(array, 0.0, 1.0)
    if rng.random() < config.blur_probability and config.blur_sigma_max:
        sigma = float(rng.uniform(0.1, config.blur_sigma_max))
        array = cv2.GaussianBlur(array, (3, 3), sigmaX=sigma, sigmaY=sigma)
    array[padding_mask] = 0.0
    output = np.rint(array * 255).astype(np.uint8)
    return cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)


def ordinal_targets(up_score: int, down_score: int) -> np.ndarray:
    """Encode two scores 0..3 as cumulative thresholds score>=1/2/3."""
    scores = np.asarray([up_score, down_score], dtype=np.int64)
    return (scores[:, None] >= np.arange(1, 4)[None, :]).astype(np.float32)


def assign_patient_folds(
    samples: Sequence[VUSample], n_folds: int = 5, seed: int = 42
) -> dict[str, int]:
    """Greedily balance up/down grade histograms without patient leakage."""
    patients: dict[str, list[VUSample]] = defaultdict(list)
    for sample in samples:
        patients[sample.patient_id].append(sample)
    if n_folds < 2 or n_folds > len(patients):
        raise ValueError("n_folds must be between 2 and the number of patients")
    vectors: dict[str, np.ndarray] = {}
    for patient, patient_samples in patients.items():
        vector = np.zeros(10, dtype=np.float64)
        for sample in patient_samples:
            vector[sample.up_score] += 1
            vector[4 + sample.down_score] += 1
            vector[8 + (sample.view == "L")] += 1
        vectors[patient] = vector
    total = np.sum(list(vectors.values()), axis=0)
    target, weights = total / n_folds, 1.0 / np.maximum(total, 1.0)
    rng = np.random.default_rng(seed)
    tie_break = {patient: float(rng.random()) for patient in patients}
    priority = {patient: float(np.sum(vector * weights)) for patient, vector in vectors.items()}
    order = sorted(patients, key=lambda patient: (-priority[patient], tie_break[patient], patient))
    fold_vectors = np.zeros((n_folds, len(target)), dtype=np.float64)
    fold_sizes = np.zeros(n_folds, dtype=np.int64)
    assignments: dict[str, int] = {}
    target_patients = len(patients) / n_folds
    for patient in order:
        costs: list[float] = []
        for fold in range(n_folds):
            candidate_vectors, candidate_sizes = fold_vectors.copy(), fold_sizes.copy()
            candidate_vectors[fold] += vectors[patient]
            candidate_sizes[fold] += 1
            distribution_cost = float(np.sum(((candidate_vectors - target) ** 2) * weights[None, :]))
            size_cost = float(np.sum((candidate_sizes - target_patients) ** 2))
            costs.append(distribution_cost + 0.1 * size_cost)
        best = min(costs)
        candidates = [fold for fold, cost in enumerate(costs) if abs(cost - best) < 1e-12]
        fold = min(candidates, key=lambda value: (fold_sizes[value], value))
        assignments[patient] = fold
        fold_vectors[fold] += vectors[patient]
        fold_sizes[fold] += 1
    return assignments


def balanced_sample_weights(samples: Sequence[VUSample], power: float = 0.5) -> np.ndarray:
    """Return mean dual-label inverse-frequency weights; ``power=.5`` is conservative."""
    if not 0 <= power <= 1:
        raise ValueError("power must be in [0, 1]")
    if not samples:
        raise ValueError("samples cannot be empty")
    up_counts = np.bincount([sample.up_score for sample in samples], minlength=4)
    down_counts = np.bincount([sample.down_score for sample in samples], minlength=4)
    if np.any(up_counts == 0) or np.any(down_counts == 0):
        raise RuntimeError("all four grades must occur in both output positions")
    count = float(len(samples))
    weights = np.asarray(
        [
            0.5
            * ((count / up_counts[sample.up_score]) ** power + (count / down_counts[sample.down_score]) ** power)
            for sample in samples
        ],
        dtype=np.float64,
    )
    return weights / weights.mean()


def build_balanced_sampler(
    samples: Sequence[VUSample], power: float = 0.5, seed: int = 42
) -> WeightedRandomSampler | None:
    """Build a reproducible sampler, or return ``None`` for natural sampling."""
    if power == 0:
        return None
    generator = torch.Generator().manual_seed(seed)
    weights = torch.as_tensor(balanced_sample_weights(samples, power), dtype=torch.double)
    return WeightedRandomSampler(weights, len(samples), replacement=True, generator=generator)


class ZhongriVUDataset(Dataset):
    """On-the-fly VU crops with train-only stochastic augmentation."""

    def __init__(
        self,
        samples: Sequence[VUSample] | None = None,
        *,
        root: Path = DEFAULT_ZHONGRI_ROOT,
        augment: bool = False,
        crop_size: int = DEFAULT_CROP_SIZE,
        augmentation: AugmentationConfig | None = None,
        seed: int = 42,
        image_cache_size: int = 4,
    ) -> None:
        self.samples = list(samples) if samples is not None else load_zhongri_samples(root)
        if not self.samples:
            raise ValueError("dataset cannot be empty")
        if crop_size < 32 or image_cache_size < 0:
            raise ValueError("crop_size must be >=32 and image_cache_size nonnegative")
        self.augment = bool(augment)
        self.crop_size = int(crop_size)
        self.augmentation = augmentation or AugmentationConfig()
        self.augmentation.validate()
        self.seed = int(seed)
        self.image_cache_size = int(image_cache_size)
        self._image_cache: OrderedDict[Path, np.ndarray] = OrderedDict()
        self._rngs: dict[int, np.random.Generator] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _rng(self) -> np.random.Generator:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else -1
        if worker_id not in self._rngs:
            worker_seed = int(torch.initial_seed() % (2**32))
            self._rngs[worker_id] = np.random.default_rng(worker_seed + self.seed)
        return self._rngs[worker_id]

    def _load_image(self, path: Path) -> np.ndarray:
        if path in self._image_cache:
            image = self._image_cache.pop(path)
            self._image_cache[path] = image
            return image
        image = decode_image(path)
        if self.image_cache_size:
            self._image_cache[path] = image
            while len(self._image_cache) > self.image_cache_size:
                self._image_cache.popitem(last=False)
        return image

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = self._load_image(sample.image_path)
        if image.shape[:2] != (sample.image_height, sample.image_width):
            raise RuntimeError(f"image shape mismatch for {sample.image_path}")
        rng = self._rng()
        jitter = sample_crop_jitter(rng, self.augmentation) if self.augment else CropJitter()
        crop, source_quad = extract_vu_crop(image, sample, self.crop_size, jitter)
        point_xy = transform_points_to_crop(
            [sample.up_corner, sample.down_corner],
            source_quad,
            self.crop_size,
        )
        if self.augment:
            crop = augment_intensity(crop, rng, self.augmentation)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        image_tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
        return {
            "image": image_tensor,
            "point_xy": torch.from_numpy(point_xy),
            "scores": torch.tensor([sample.up_score, sample.down_score], dtype=torch.long),
            "ordinal_targets": torch.from_numpy(ordinal_targets(sample.up_score, sample.down_score)),
            "view_id": torch.tensor(0 if sample.view == "C" else 1, dtype=torch.long),
            "level_index": torch.tensor(sample.level_index - 1, dtype=torch.long),
            "sample_id": sample.sample_id,
            "patient_id": sample.patient_id,
            "view": sample.view,
            "level_name": sample.level_name,
            "source_quad": torch.from_numpy(source_quad),
            "jitter": torch.from_numpy(jitter.as_array()),
        }


def select_samples(samples: Iterable[VUSample], patient_ids: set[str]) -> list[VUSample]:
    """Select whole-patient subsets for leakage-free loaders."""
    return [sample for sample in samples if sample.patient_id in patient_ids]
