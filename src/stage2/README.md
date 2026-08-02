# Stage 2: VU up/down scoring data

Stage 2 uses only the 116 scored Zhongri radiographs. Each image contributes
six oriented VU crops, and every crop keeps two independent integer labels:

- `up`: the spatially superior corner score, 0–3;
- `down`: the spatially inferior corner score, 0–3.

CSXA and BUU-LSPINE are not inserted into this supervised index because they
do not have mSASSS scores. In particular, missing scores are never converted
to grade 0.

## Prepare the index and preview

```bash
cd /home/lsw/lv/mSASSS_pipeline
bash src/stage2/prepare_data.sh --exist-ok
```

Outputs are written to `src/stage2/data/zhongri_vu/`:

- `manifest.csv`: 696 VUs with patient, view, level, up/down scores, and fold;
- `summary.json`: grade distributions overall/by view/by level/by fold;
- `augmentation_preview.jpg`: canonical crops beside three random variants.

The five folds are assigned at patient level. Both the cervical and lumbar
image of a patient always stay in the same fold.

## Crop convention

The adjacent Stage-1 vertebral centers form the posterior crop edge. The crop
extends toward the two annotated VU corners by one center distance and is
unwarped to 256×256. Every crop therefore has the same anatomy orientation:

```text
left = anterior, right = posterior
top  = superior, bottom = inferior
```

No horizontal or vertical flip is allowed because either would change this
meaning and could exchange the up/down endpoints.

## Online training augmentation

Augmentation is generated anew in `ZhongriVUDataset(..., augment=True)` on
every access; images are not copied into a finite augmented dataset.

- crop rotation: ±3°;
- crop translation: ±4% in each crop axis;
- field-of-view scale: 0.90–1.10;
- gamma and contrast: ±10%;
- brightness: ±0.03 in normalized intensity;
- light grayscale Gaussian noise with 25% probability;
- light 3×3 Gaussian blur with 15% probability.

Validation/test loaders must use `augment=False`. Flips, MixUp, CutMix,
random erasing, elastic deformation, and pseudo-negative grade 0 labels are
explicitly excluded.

For the grade imbalance, `build_balanced_sampler(train_samples, power=0.5)`
provides square-root inverse-frequency sampling. It weighs `up` and `down`
separately and averages the two contributions rather than collapsing each VU
into one artificial class. Natural sampling (`power=0`) remains the baseline.

## Train EfficientNet-B0 ordinal scorer

The Stage-2 model receives one online-augmented `256 x 256` VU and produces
two monotonic ordinal outputs. Each output has three cumulative decisions:
`score >= 1`, `score >= 2`, and `score >= 3`. The validation split is one
patient-level fold and is never augmented.

Run one fold:

```bash
cd /home/lsw/lv/mSASSS_pipeline
bash src/stage2/train_efficientnet.sh --fold 0
```

Run all five folds sequentially:

```bash
bash src/stage2/train_all_folds.sh
```

Defaults are ImageNet initialization, 150 epochs, batch 32, five frozen-backbone
epochs, square-root grade balancing, and early stopping after 30 epochs without
validation mean-MAE improvement. Outputs are stored under
`src/stage2/outputs/efficientnet/fold_N/`.

To initialize from a leak-safe Stage-1 EfficientNet checkpoint instead:

```bash
bash src/stage2/train_efficientnet.sh \
  --fold 0 \
  --stage1-weights /path/to/stage1/weights/best.pt
```

Only the Stage-1 `backbone` is loaded; its heatmap/FPN/classification heads are
discarded. A Stage-1 checkpoint that has seen the Stage-2 validation patients
must not be used for a strict cross-validation result.
