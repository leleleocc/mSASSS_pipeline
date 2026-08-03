# Stage 1: 19-point landmark localization

This directory is the permanent home of Stage-1 code and its prepared 19-point dataset.
Source files live directly in this directory; there is no second nested source directory.

## Layout

```text
src/stage1/
├── data/spine_keypoints_19pt/   # prepared dataset
├── efficientnet.py              # shared B0 model/data/loss implementation
├── yolo_strategy.py             # fixed-shape YOLO dataset, structure loss, N-MRE validator
├── train_*.py / evaluate_*.py
├── train_*.sh / evaluate_*.sh
└── outputs/                     # created by training/evaluation
```

Both models use a fixed `1024 x 768` tensor in `(height, width)` order. Images keep
their aspect ratio and are padded; they are not stretched.

`prepare_data.py` is the single canonical builder for the Zhongri + CSXA +
BUU-LSPINE 19-point dataset. Use `prepare_data.sh --check-only` for a read-only audit.

Detailed implementation documents:

- [YOLO26s-Pose augmentation, architecture, and training](docs/yolo.md)
- [EfficientNet-B0-FPN augmentation, architecture, and training](docs/efficientnet.md)

## Commands

From any directory:

```bash
bash /home/lsw/lv/mSASSS_pipeline/src/stage1/train_yolo_pose.sh --help
bash /home/lsw/lv/mSASSS_pipeline/src/stage1/train_efficientnet.sh --help
```

Full runs:

```bash
bash src/stage1/train_yolo_pose.sh
bash src/stage1/train_efficientnet.sh
```

The default geometric augmentation is `5 degrees / 3% translation / 10% scale`.
YOLO optimizes box, pose, keypoint-objectness, class/distribution terms, and the
additional subject-specific spine structure loss. Its `best.pt` is selected by the
lowest validation normalized MRE. EfficientNet uses heatmap, coordinate, structure,
and auxiliary cervical/lumbar classification losses and also selects `best.pt` by
validation normalized MRE.

Evaluation examples:

```bash
bash src/stage1/evaluate_yolo_pose.sh WEIGHTS --split val --name yolo_val
bash src/stage1/evaluate_efficientnet.sh WEIGHTS --split val --name efficientnet_val
```
