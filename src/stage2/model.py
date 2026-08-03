"""EfficientNet-B0 dual-ordinal model for VU up/down scoring."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from torchvision.ops import roi_align


NUM_ENDPOINTS = 2
NUM_THRESHOLDS = 3
ENDPOINT_NAMES = ("up", "down")


class MonotonicOrdinalHead(nn.Module):
    """CORAL-style endpoint scores with learned ordered thresholds."""

    def __init__(self, input_dim: int, endpoints: int = NUM_ENDPOINTS) -> None:
        super().__init__()
        self.endpoints = int(endpoints)
        self.score = nn.Linear(input_dim, self.endpoints)
        self.threshold_start = nn.Parameter(torch.full((self.endpoints, 1), -1.0))
        initial_gap = math.log(math.expm1(1.0))
        self.threshold_gaps = nn.Parameter(torch.full((self.endpoints, NUM_THRESHOLDS - 1), initial_gap))

    def thresholds(self) -> torch.Tensor:
        gaps = F.softplus(self.threshold_gaps)
        return torch.cat((self.threshold_start, self.threshold_start + gaps.cumsum(dim=1)), dim=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        latent = self.score(features).unsqueeze(-1)
        return latent - self.thresholds().unsqueeze(0)


class EndpointMonotonicOrdinalHead(nn.Module):
    """Monotonic ordinal scores from one feature vector per endpoint."""

    def __init__(self, input_dim: int, endpoints: int = NUM_ENDPOINTS) -> None:
        super().__init__()
        self.endpoints = int(endpoints)
        self.score = nn.ModuleList(nn.Linear(input_dim, 1) for _ in range(self.endpoints))
        self.threshold_start = nn.Parameter(torch.full((self.endpoints, 1), -1.0))
        initial_gap = math.log(math.expm1(1.0))
        self.threshold_gaps = nn.Parameter(
            torch.full((self.endpoints, NUM_THRESHOLDS - 1), initial_gap)
        )

    def thresholds(self) -> torch.Tensor:
        gaps = F.softplus(self.threshold_gaps)
        return torch.cat(
            (self.threshold_start, self.threshold_start + gaps.cumsum(dim=1)),
            dim=1,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[1] != self.endpoints:
            raise ValueError(
                f"expected endpoint features [B,{self.endpoints},D], got {tuple(features.shape)}"
            )
        latent = torch.stack(
            [layer(features[:, endpoint]).squeeze(-1) for endpoint, layer in enumerate(self.score)],
            dim=1,
        )
        return latent.unsqueeze(-1) - self.thresholds().unsqueeze(0)


def build_endpoint_roi_boxes(point_xy: torch.Tensor, roi_size: float) -> torch.Tensor:
    """Build ROIAlign boxes ordered as up/down for every batch item."""
    if point_xy.ndim != 3 or point_xy.shape[1:] != (NUM_ENDPOINTS, 2):
        raise ValueError(f"expected point_xy [B,2,2], got {tuple(point_xy.shape)}")
    if roi_size <= 0:
        raise ValueError("roi_size must be positive")
    if not torch.isfinite(point_xy).all():
        raise ValueError("point_xy contains non-finite coordinates")
    batch_size = point_xy.shape[0]
    centers = point_xy.reshape(-1, 2)
    half = float(roi_size) / 2.0
    corners = torch.cat((centers - half, centers + half), dim=1)
    batch_indices = (
        torch.arange(batch_size, device=point_xy.device, dtype=point_xy.dtype)
        .unsqueeze(1)
        .expand(batch_size, NUM_ENDPOINTS)
        .reshape(-1, 1)
    )
    return torch.cat((batch_indices, corners), dim=1)


class VUOrdinalEfficientNet(nn.Module):
    """Global EfficientNet-B0 scorer with optional point-guided endpoint ROIs."""

    def __init__(
        self,
        *,
        pretrained: bool = True,
        hidden_dim: int = 256,
        dropout: float = 0.30,
        use_roi: bool = False,
        local_roi_size: float = 64.0,
        roi_output_size: int = 5,
        local_dim: int = 128,
        local_dropout: float = 0.20,
    ) -> None:
        super().__init__()
        if hidden_dim < 32:
            raise ValueError("hidden_dim must be at least 32")
        if local_dim < 16:
            raise ValueError("local_dim must be at least 16")
        if not 0 <= dropout < 1 or not 0 <= local_dropout < 1:
            raise ValueError("dropout values must be in [0, 1)")
        if local_roi_size <= 0 or roi_output_size < 1:
            raise ValueError("local_roi_size and roi_output_size must be positive")
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        base = efficientnet_b0(weights=weights)
        self.use_roi = bool(use_roi)
        self.local_roi_size = float(local_roi_size)
        self.roi_output_size = int(roi_output_size)
        self.backbone = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.neck = nn.Sequential(
            nn.Linear(1280, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )
        if self.use_roi:
            self.local_pool = nn.AdaptiveAvgPool2d(1)
            self.local_projection = nn.Sequential(
                nn.Linear(40, local_dim),
                nn.LayerNorm(local_dim),
                nn.SiLU(inplace=True),
                nn.Dropout(local_dropout),
            )
            self.endpoint_fusion = nn.Sequential(
                nn.Linear(hidden_dim + 2 * local_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(inplace=True),
            )
            self.ordinal_head: nn.Module = EndpointMonotonicOrdinalHead(hidden_dim)
        else:
            self.local_pool = None
            self.local_projection = None
            self.endpoint_fusion = None
            self.ordinal_head = MonotonicOrdinalHead(hidden_dim)

    def _backbone_features(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        features = image
        stride8 = None
        for index, block in enumerate(self.backbone):
            features = block(features)
            if self.use_roi and index == 3:
                stride8 = features
        if self.use_roi and stride8 is None:
            raise RuntimeError("EfficientNet-B0 stride-8 feature map was not produced")
        return features, stride8

    def _endpoint_embeddings(
        self,
        image: torch.Tensor,
        stride8: torch.Tensor,
        global_embedding: torch.Tensor,
        point_xy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.local_pool is None or self.local_projection is None or self.endpoint_fusion is None:
            raise RuntimeError("ROI modules are unavailable when use_roi=False")
        if image.shape[-2] != image.shape[-1]:
            raise ValueError("ROI mode requires square canonical VU inputs")
        if point_xy.shape[0] != image.shape[0]:
            raise ValueError(
                f"point_xy batch {point_xy.shape[0]} does not match image batch {image.shape[0]}"
            )
        point_xy = point_xy.to(device=stride8.device, dtype=stride8.dtype)
        boxes = build_endpoint_roi_boxes(point_xy, self.local_roi_size)
        spatial_scale = float(stride8.shape[-1]) / float(image.shape[-1])
        local_features = roi_align(
            stride8,
            boxes,
            output_size=(self.roi_output_size, self.roi_output_size),
            spatial_scale=spatial_scale,
            sampling_ratio=2,
            aligned=True,
        )
        batch_size = image.shape[0]
        local_embedding = self.local_pool(local_features).flatten(1)
        local_embedding = self.local_projection(local_embedding).reshape(
            batch_size, NUM_ENDPOINTS, -1
        )
        global_per_endpoint = global_embedding.unsqueeze(1).expand(-1, NUM_ENDPOINTS, -1)
        primary_local = local_embedding
        counterpart_local = local_embedding.flip(dims=(1,))
        fusion_input = torch.cat(
            (global_per_endpoint, primary_local, counterpart_local),
            dim=-1,
        )
        return self.endpoint_fusion(fusion_input), local_embedding, boxes

    def forward(
        self,
        image: torch.Tensor,
        point_xy: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        features, stride8 = self._backbone_features(image)
        pooled = self.pool(features).flatten(1)
        global_embedding = self.neck(pooled)
        local_embedding = None
        roi_boxes = None
        if self.use_roi:
            if point_xy is None:
                raise ValueError("point_xy is required when use_roi=True")
            if stride8 is None:
                raise RuntimeError("missing stride-8 features")
            embedding, local_embedding, roi_boxes = self._endpoint_embeddings(
                image,
                stride8,
                global_embedding,
                point_xy,
            )
        else:
            embedding = global_embedding
        logits = self.ordinal_head(embedding)
        outputs = {
            "ordinal_logits": logits,
            "probabilities": logits.sigmoid(),
            "scores": decode_scores(logits),
            "embedding": embedding,
            "global_embedding": global_embedding,
        }
        if local_embedding is not None and roi_boxes is not None:
            outputs["local_embedding"] = local_embedding
            outputs["roi_boxes"] = roi_boxes
        return outputs


def decode_scores(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Decode cumulative probabilities into integer scores 0..3."""
    if logits.ndim != 3 or logits.shape[1:] != (NUM_ENDPOINTS, NUM_THRESHOLDS):
        raise ValueError(f"expected logits [B,2,3], got {tuple(logits.shape)}")
    return (logits.sigmoid() >= threshold).sum(dim=-1).long()


def ordinal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    endpoint_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Binary cross-entropy over the six cumulative up/down decisions."""
    if logits.shape != targets.shape or logits.ndim != 3 or logits.shape[1:] != (2, 3):
        raise ValueError(f"expected matching [B,2,3] tensors, got {tuple(logits.shape)} and {tuple(targets.shape)}")
    per_element = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
    per_endpoint = per_element.mean(dim=(0, 2))
    if endpoint_weights is None:
        total = per_endpoint.mean()
    else:
        weights = endpoint_weights.to(device=logits.device, dtype=logits.dtype)
        if weights.shape != (2,) or torch.any(weights <= 0):
            raise ValueError("endpoint_weights must contain two positive values")
        total = (per_endpoint * weights).sum() / weights.sum()
    return total, {"up_loss": per_endpoint[0], "down_loss": per_endpoint[1]}


def _checkpoint_state(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Stage-1 checkpoint must be a dictionary")
    state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict) or not state:
        raise RuntimeError("Stage-1 checkpoint does not contain a model state")
    return state


def load_stage1_backbone(model: VUOrdinalEfficientNet, checkpoint_path: Path) -> dict[str, Any]:
    """Load only ``EfficientNetKeypointModel.backbone`` from a Stage-1 checkpoint."""
    path = checkpoint_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = _checkpoint_state(checkpoint)
    backbone_state: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        normalized = str(key)
        for prefix in ("module.", "_orig_mod."):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
        if normalized.startswith("backbone."):
            backbone_state[normalized.removeprefix("backbone.")] = value
    if not backbone_state:
        raise RuntimeError(f"no EfficientNet backbone parameters found in {path}")
    missing, unexpected = model.backbone.load_state_dict(backbone_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Stage-1 backbone is incompatible: missing={list(missing)}, unexpected={list(unexpected)}"
        )
    return {
        "path": str(path),
        "checkpoint_epoch": int(checkpoint["epoch"]) + 1 if "epoch" in checkpoint else None,
        "parameters_loaded": len(backbone_state),
    }


def trainable_parameter_counts(model: nn.Module) -> dict[str, int]:
    """Return total/backbone/head parameter counts for logging."""
    backbone = sum(parameter.numel() for parameter in model.backbone.parameters())
    total = sum(parameter.numel() for parameter in model.parameters())
    return {"total": total, "backbone": backbone, "head": total - backbone}
