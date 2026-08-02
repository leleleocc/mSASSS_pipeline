"""EfficientNet-B0 dual-ordinal model for VU up/down scoring."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


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


class VUOrdinalEfficientNet(nn.Module):
    """EfficientNet-B0 backbone with one monotonic ordinal head for up/down."""

    def __init__(self, *, pretrained: bool = True, hidden_dim: int = 256, dropout: float = 0.30) -> None:
        super().__init__()
        if hidden_dim < 32:
            raise ValueError("hidden_dim must be at least 32")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        base = efficientnet_b0(weights=weights)
        self.backbone = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.neck = nn.Sequential(
            nn.Linear(1280, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.ordinal_head = MonotonicOrdinalHead(hidden_dim)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(image)
        pooled = self.pool(features).flatten(1)
        embedding = self.neck(pooled)
        logits = self.ordinal_head(embedding)
        return {
            "ordinal_logits": logits,
            "probabilities": logits.sigmoid(),
            "scores": decode_scores(logits),
            "embedding": embedding,
        }


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
