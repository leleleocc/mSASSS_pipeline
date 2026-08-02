from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from src.stage1.efficientnet import (
    EfficientNetKeypointModel,
    keypoint_loss,
    letterbox,
    spine_structure_loss,
)
from src.stage1.yolo_strategy import _spine_structure_loss, normalize_input_shape


class Stage1StrategyTest(unittest.TestCase):
    def test_fixed_rectangular_letterbox(self) -> None:
        image = np.full((200, 100, 3), 127, dtype=np.uint8)
        points = np.asarray([[20, 40], [70, 160]], dtype=np.float32)
        output, transformed, scale, pad = letterbox(image, points, (1024, 768))
        self.assertEqual(output.shape, (1024, 768, 3))
        self.assertGreater(scale, 0)
        self.assertEqual(len(pad), 2)
        self.assertTrue(np.all(transformed >= 0))
        self.assertEqual(normalize_input_shape([1024, 768]), (1024, 768))

    def test_efficientnet_forward_and_composite_loss(self) -> None:
        model = EfficientNetKeypointModel(pretrained=False).eval()
        image = torch.zeros(1, 3, 128, 96)
        target = torch.linspace(0.1, 0.9, 19).view(1, 19, 1).repeat(1, 1, 2)
        with torch.no_grad():
            outputs = model(image)
            loss, parts = keypoint_loss(outputs, target, torch.zeros(1, dtype=torch.long))
        self.assertEqual(outputs["coordinates"].shape, (1, 19, 2))
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("structure_loss", parts)

    def test_structure_losses_are_differentiable(self) -> None:
        target = torch.zeros(2, 19, 2)
        target[..., 1] = torch.arange(19).float() / 20
        predicted = (target + 0.01 * torch.randn_like(target)).requires_grad_()
        efficientnet_loss, _ = spine_structure_loss(predicted, target)
        yolo_loss = _spine_structure_loss(predicted, target)
        total = efficientnet_loss + yolo_loss
        total.backward()
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(torch.isfinite(predicted.grad).all())

    def test_dataset_yaml_resolves_relative_to_yaml_directory(self) -> None:
        data = Path(__file__).resolve().parents[1] / "data/spine_keypoints_19pt/data.yaml"
        self.assertTrue(data.is_file())
        text = data.read_text(encoding="utf-8")
        config = yaml.safe_load(text)
        root = Path(config.get("path", data.parent))
        if not root.is_absolute():
            root = (data.parent / root).resolve()
        self.assertEqual(root, data.parent.resolve())
        self.assertNotIn("Ortho_Qwen", text)


if __name__ == "__main__":
    unittest.main()
