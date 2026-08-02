"""Fixed-shape YOLO26-Pose training with spine geometry loss and N-MRE selection."""

from __future__ import annotations

import os
from copy import copy
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F


STAGE1_ROOT = Path(__file__).resolve().parent
MPL_CONFIG = STAGE1_ROOT / ".matplotlib"
MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG))

from ultralytics.data.augment import (
    Albumentations,
    Compose,
    Format,
    LetterBox,
    RandomHSV,
    RandomPerspective,
)
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.pose.train import PoseTrainer
from ultralytics.models.yolo.pose.val import PoseValidator
from ultralytics.nn.modules.head import Pose26
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils import DEFAULT_CFG, RANK, ops
from ultralytics.utils.loss import E2ELoss, PoseLoss26, v8PoseLoss
from ultralytics.utils.torch_utils import unwrap_model


NUM_KEYPOINTS = 19
DEFAULT_INPUT_SHAPE = (1024, 768)  # height, width


def normalize_input_shape(value: int | Sequence[int]) -> tuple[int, int]:
    """Normalize one square size or ``height width`` to a stride-safe shape."""
    values = [int(value)] if isinstance(value, int) else [int(item) for item in value]
    if len(values) == 1:
        shape = (values[0], values[0])
    elif len(values) == 2:
        shape = (values[0], values[1])
    else:
        raise ValueError("imgsz must contain one value or height width")
    if any(item < 64 or item % 32 for item in shape):
        raise ValueError("each imgsz dimension must be >=64 and divisible by 32")
    return shape


def _spine_structure_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match subject-specific adjacent vectors, VU pairs, and vertical ordering."""
    if predicted.shape != target.shape or predicted.shape[-2:] != (NUM_KEYPOINTS, 2):
        raise ValueError(f"expected matching [N,{NUM_KEYPOINTS},2] tensors")
    chains = (
        (0, 1, 2, 3, 4, 5, 6),
        (7, 9, 11, 13, 15, 17),
        (8, 10, 12, 14, 16, 18),
    )
    pred_vectors: list[torch.Tensor] = []
    target_vectors: list[torch.Tensor] = []
    order_terms: list[torch.Tensor] = []
    for chain in chains:
        index = torch.as_tensor(chain, device=predicted.device)
        pred_delta = predicted.index_select(1, index)[:, 1:] - predicted.index_select(1, index)[:, :-1]
        target_delta = target.index_select(1, index)[:, 1:] - target.index_select(1, index)[:, :-1]
        pred_vectors.append(pred_delta)
        target_vectors.append(target_delta)
        direction = torch.where(target_delta[..., 1] >= 0, 1.0, -1.0)
        order_terms.append(F.relu(0.002 - pred_delta[..., 1] * direction))

    adjacent = F.smooth_l1_loss(
        torch.cat(pred_vectors, dim=1), torch.cat(target_vectors, dim=1), beta=0.01
    )
    p1 = torch.as_tensor((7, 9, 11, 13, 15, 17), device=predicted.device)
    p2 = torch.as_tensor((8, 10, 12, 14, 16, 18), device=predicted.device)
    pred_pairs = predicted.index_select(1, p2) - predicted.index_select(1, p1)
    target_pairs = target.index_select(1, p2) - target.index_select(1, p1)
    pairs = F.smooth_l1_loss(pred_pairs, target_pairs, beta=0.01)
    order = torch.cat(order_terms, dim=1).mean()
    return 0.5 * adjacent + 0.3 * pairs + 0.2 * order


class SpinePoseLoss(PoseLoss26):
    """YOLO26 pose loss with an additional differentiable 19-point structure term."""

    def __init__(self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None):
        super().__init__(model, tal_topk, tal_topk2)
        self.structure_gain = float(getattr(model, "spine_structure_gain", 1.0))
        self.loss_names += ("structure_loss",)
        self._structure = torch.zeros((), device=self.device)

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._structure = pred_kpts[..., :2].sum() * 0.0
        if masks.any():
            selected = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)
            per_anchor_stride = stride_tensor.view(1, -1).expand(masks.shape[0], -1)[masks]
            gt = selected[masks]
            gt_xy = gt[..., :2] / per_anchor_stride[:, None, None]
            pred_xy = pred_kpts[masks][..., :2]
            boxes = target_bboxes[masks] / per_anchor_stride[:, None]
            origin = boxes[:, None, :2]
            size = (boxes[:, 2:] - boxes[:, :2]).clamp_min(1.0)[:, None, :]
            self._structure = _spine_structure_loss(
                (pred_xy - origin) / size,
                (gt_xy - origin) / size,
            )
        return super().calculate_keypoints_loss(
            masks,
            target_gt_idx,
            keypoints,
            batch_idx,
            stride_tensor,
            target_bboxes,
            pred_kpts,
        )

    def loss(
        self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        components, items = super().loss(preds, batch)
        weighted = self._structure * self.structure_gain
        components = torch.cat((components, weighted.reshape(1) * preds["kpts"].shape[0]))
        items["structure_loss"] = weighted.detach()
        return components, items


class SpinePoseModel(PoseModel):
    """PoseModel that installs :class:`SpinePoseLoss` for YOLO26 heads."""

    def init_criterion(self):
        if self.end2end or isinstance(self.model[-1], Pose26):
            return E2ELoss(self, SpinePoseLoss) if self.end2end else SpinePoseLoss(self)
        return v8PoseLoss(self)


class FixedShapePoseDataset(YOLODataset):
    """YOLO pose dataset whose final tensor is always ``height x width``."""

    def __init__(self, *args, input_shape: Sequence[int] = DEFAULT_INPUT_SHAPE, **kwargs):
        self.input_shape = normalize_input_shape(input_shape)
        super().__init__(*args, **kwargs)

    def build_transforms(self, hyp=None) -> Compose:
        height, width = self.input_shape
        if self.augment:
            transforms = Compose(
                [
                    RandomPerspective(
                        degrees=hyp.degrees,
                        translate=hyp.translate,
                        scale=hyp.scale,
                        shear=hyp.shear,
                        perspective=hyp.perspective,
                        size=(width, height),
                    ),
                    Albumentations(p=1.0, transforms=getattr(hyp, "augmentations", None)),
                    RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
                ]
            )
        else:
            transforms = Compose([LetterBox(new_shape=(height, width), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,
            )
        )
        return transforms


def _build_fixed_dataset(owner: Any, img_path: str, mode: str, batch: int | None):
    stride = max(int(unwrap_model(owner.model).stride.max()), 32)
    return FixedShapePoseDataset(
        img_path=img_path,
        imgsz=max(owner.input_shape),
        batch_size=batch,
        augment=mode == "train",
        hyp=owner.args,
        rect=False,
        cache=owner.args.cache or None,
        single_cls=owner.args.single_cls or False,
        stride=stride,
        pad=0.0,
        prefix=f"{mode}: ",
        task=owner.args.task,
        classes=owner.args.classes,
        data=owner.data,
        fraction=owner.args.fraction if mode == "train" else 1.0,
        input_shape=owner.input_shape,
    )


class NormalizedMREPoseValidator(PoseValidator):
    """Add original-image N-MRE/PCK and use negative N-MRE as fitness."""

    def __init__(self, *args, input_shape: Sequence[int] = DEFAULT_INPUT_SHAPE, **kwargs):
        self.input_shape = normalize_input_shape(input_shape)
        self._nme_sum = 0.0
        self._pck1_count = 0.0
        self._point_count = 0.0
        super().__init__(*args, **kwargs)

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None):
        return _build_fixed_dataset(self, img_path, mode, batch)

    def init_metrics(self, model: torch.nn.Module) -> None:
        self._nme_sum = self._pck1_count = self._point_count = 0.0
        super().init_metrics(model)

    def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
        for index, prediction in enumerate(preds):
            prepared = self._prepare_batch(index, batch)
            target = prepared["keypoints"]
            if not len(target):
                continue
            original_shape = prepared["ori_shape"]
            diagonal = float(np.hypot(*original_shape))
            target_xy = ops.scale_coords(
                prepared["imgsz"],
                target[..., :2].clone(),
                original_shape,
                ratio_pad=prepared["ratio_pad"],
            )[0]
            if prediction["conf"].numel():
                best = int(prediction["conf"].argmax())
                predicted_xy = self.scale_preds(
                    {key: value.clone() for key, value in prediction.items()}, prepared
                )["keypoints"][best, :, :2]
                normalized = torch.linalg.vector_norm(predicted_xy - target_xy, dim=-1) / diagonal
            else:
                normalized = torch.ones(target_xy.shape[0], device=target_xy.device)
            self._nme_sum += float(normalized.sum().detach().cpu())
            self._pck1_count += float((normalized <= 0.01).sum().detach().cpu())
            self._point_count += float(normalized.numel())
        super().update_metrics(preds, batch)

    def get_stats(self) -> dict[str, Any]:
        stats = super().get_stats()
        totals = torch.tensor(
            [self._nme_sum, self._pck1_count, self._point_count],
            dtype=torch.float64,
            device=self.device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        nme_sum, pck_count, point_count = totals.tolist()
        denominator = max(point_count, 1.0)
        nme = nme_sum / denominator
        stats["metrics/normalized_MRE(P)"] = nme * 100.0
        stats["metrics/PCK@1%(P)"] = pck_count / denominator
        stats["fitness"] = -nme
        return stats


class SpinePoseTrainer(PoseTrainer):
    """Trainer joining fixed 1024x768 inputs, structure loss, and N-MRE selection."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        overrides = dict(overrides or {})
        self.input_shape = normalize_input_shape(overrides.pop("input_shape", DEFAULT_INPUT_SHAPE))
        self.structure_gain = float(overrides.pop("spine_structure_gain", 1.0))
        overrides["imgsz"] = max(self.input_shape)
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True) -> SpinePoseModel:
        model = self.set_model_names_for_load(
            SpinePoseModel(
                cfg,
                nc=self.data["nc"],
                ch=self.data["channels"],
                data_kpt_shape=self.data["kpt_shape"],
                verbose=verbose and RANK == -1,
            )
        )
        model.spine_structure_gain = self.structure_gain
        model.stage1_input_shape = self.input_shape
        if weights:
            model.load(weights)
        return model

    def set_model_attributes(self):
        super().set_model_attributes()
        self.model.spine_structure_gain = self.structure_gain
        self.model.stage1_input_shape = self.input_shape

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        return _build_fixed_dataset(self, img_path, mode, batch)

    def get_validator(self):
        return NormalizedMREPoseValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
            input_shape=self.input_shape,
        )
