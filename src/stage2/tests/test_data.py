from __future__ import annotations

import unittest
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import torch

from src.stage2.data import (
    AugmentationConfig,
    CropJitter,
    ZhongriVUDataset,
    assign_patient_folds,
    augment_intensity,
    balanced_sample_weights,
    crop_quad,
    decode_image,
    extract_vu_crop,
    load_zhongri_samples,
    ordinal_targets,
    sample_crop_jitter,
)
from src.stage2.model import (
    VUOrdinalEfficientNet,
    load_stage1_backbone,
    ordinal_loss,
    trainable_parameter_counts,
)
from src.stage2.train_efficientnet import quadratic_weighted_kappa


ROOT = Path(__file__).resolve().parents[3]


class Stage2DataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.samples = load_zhongri_samples()

    def test_all_scored_samples_are_indexed(self) -> None:
        self.assertEqual(len(self.samples), 696)
        self.assertEqual(len({sample.patient_id for sample in self.samples}), 58)
        self.assertEqual(len({sample.image_path for sample in self.samples}), 116)
        self.assertEqual(Counter(sample.up_score for sample in self.samples), {0: 37, 1: 294, 2: 266, 3: 99})
        self.assertEqual(Counter(sample.down_score for sample in self.samples), {0: 71, 1: 351, 2: 178, 3: 96})

    def test_canonical_crop_reproduces_existing_reference(self) -> None:
        sample = next(item for item in self.samples if item.sample_id == "G016-C__V3")
        crop, _ = extract_vu_crop(decode_image(sample.image_path), sample, 256, CropJitter())
        reference = cv2.imread(
            str(ROOT / "raw_data/zhongri/0-before-trim/output_subimgs/G016-C/sub_3.png")
        )
        self.assertIsNotNone(reference)
        difference = np.abs(crop.astype(np.float32) - reference.astype(np.float32)).mean()
        self.assertLess(difference, 2.0)

    def test_dataset_returns_two_scores_and_ordinal_targets(self) -> None:
        item = ZhongriVUDataset(self.samples[:2], augment=False)[0]
        self.assertEqual(tuple(item["image"].shape), (3, 256, 256))
        self.assertEqual(tuple(item["scores"].shape), (2,))
        self.assertEqual(tuple(item["ordinal_targets"].shape), (2, 3))
        np.testing.assert_array_equal(
            ordinal_targets(2, 1), np.asarray([[1, 1, 0], [1, 0, 0]], dtype=np.float32)
        )

    def test_jitter_and_intensity_are_bounded_and_grayscale(self) -> None:
        config = AugmentationConfig()
        rng = np.random.default_rng(42)
        sample = self.samples[0]
        canonical = crop_quad(sample)
        for _ in range(100):
            jitter = sample_crop_jitter(rng, config)
            self.assertLessEqual(abs(jitter.rotation_deg), config.rotation_deg)
            self.assertLessEqual(abs(jitter.translate_x_fraction), config.translation_fraction)
            self.assertLessEqual(abs(jitter.translate_y_fraction), config.translation_fraction)
            self.assertGreaterEqual(jitter.field_scale, 1 - config.field_scale)
            self.assertLessEqual(jitter.field_scale, 1 + config.field_scale)
        crop, _ = extract_vu_crop(decode_image(sample.image_path), sample, 256)
        augmented = augment_intensity(crop, rng, config)
        np.testing.assert_array_equal(augmented[..., 0], augmented[..., 1])
        np.testing.assert_array_equal(augmented[..., 1], augmented[..., 2])
        self.assertEqual(canonical.shape, (4, 2))

    def test_patient_folds_have_no_cross_fold_leakage(self) -> None:
        assignments = assign_patient_folds(self.samples, 5, 42)
        by_fold: dict[int, set[str]] = defaultdict(set)
        for patient, fold in assignments.items():
            by_fold[fold].add(patient)
        self.assertEqual(sorted(len(value) for value in by_fold.values()), [11, 11, 12, 12, 12])
        self.assertEqual(sum(len(value) for value in by_fold.values()), 58)
        self.assertTrue(all(assignments[sample.patient_id] in range(5) for sample in self.samples))

    def test_sqrt_balancing_upweights_rare_dual_labels(self) -> None:
        weights = balanced_sample_weights(self.samples, power=0.5)
        rare = np.mean(
            [weight for sample, weight in zip(self.samples, weights, strict=True) if sample.up_score == 0]
        )
        common = np.mean(
            [weight for sample, weight in zip(self.samples, weights, strict=True) if sample.up_score == 1]
        )
        self.assertGreater(rare, common)

    def test_ordinal_model_is_monotonic_and_loss_is_differentiable(self) -> None:
        model = VUOrdinalEfficientNet(pretrained=False, hidden_dim=64).eval()
        with torch.no_grad():
            output = model(torch.zeros(2, 3, 64, 64))
        self.assertEqual(tuple(output["ordinal_logits"].shape), (2, 2, 3))
        probabilities = output["probabilities"]
        self.assertTrue(bool((probabilities[..., 0] >= probabilities[..., 1]).all()))
        self.assertTrue(bool((probabilities[..., 1] >= probabilities[..., 2]).all()))
        logits = torch.randn(4, 2, 3, requires_grad=True)
        targets = torch.from_numpy(np.stack([ordinal_targets(0, 3), ordinal_targets(2, 1)] * 2))
        loss, parts = ordinal_loss(logits, targets)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(set(parts), {"up_loss", "down_loss"})
        self.assertGreater(trainable_parameter_counts(model)["backbone"], 1_000_000)

    def test_stage1_efficientnet_backbone_can_be_loaded(self) -> None:
        from src.stage1.efficientnet import EfficientNetKeypointModel

        stage1 = EfficientNetKeypointModel(pretrained=False)
        stage2 = VUOrdinalEfficientNet(pretrained=False, hidden_dim=64)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "stage1.pt"
            torch.save({"model_state": stage1.state_dict(), "epoch": 0}, path)
            metadata = load_stage1_backbone(stage2, path)
        self.assertEqual(metadata["checkpoint_epoch"], 1)
        self.assertGreater(metadata["parameters_loaded"], 0)

    def test_quadratic_kappa_is_one_for_perfect_predictions(self) -> None:
        confusion = np.diag([2, 3, 4, 5])
        self.assertAlmostEqual(quadratic_weighted_kappa(confusion), 1.0)


if __name__ == "__main__":
    unittest.main()
