# Stage 2：VU up/down 评分

Stage 2 只使用 Zhongri 中带评分的 116 张 X 光片。每张图生成 6 个方向统一的 VU
crop，每个 crop 保留两个相互独立的整数标签：

- `up`：空间上方评分角的分数，范围为 0–3；
- `down`：空间下方评分角的分数，范围为 0–3。

CSXA 和 BUU-LSPINE 没有 mSASSS 分数，因此不会加入 Stage 2 的监督训练索引。
缺失分数不能转换为 0 分。

详细设计文档：

- [EfficientNet up/down ROI 架构与训练策略](docs/efficientnet.md)

Stage 2 不使用 YOLO。当前代码同时支持全局 EfficientNet-B0 基线和可选的 point-aware
模型：由 Stage 1 的 up/down 关键点驱动两个局部 ROIAlign 分支，同时使用完整 VU
分支保留椎间上下文。

## 准备数据索引和预览图

```bash
cd /home/lsw/lv/mSASSS_pipeline
bash src/stage2/prepare_data.sh --exist-ok
```

输出保存在 `src/stage2/data/zhongri_vu/`：

- `manifest.csv`：696 个 VU，包含患者、视图、椎间级别、up/down 分数和 fold；
- `summary.json`：总体及按视图、级别、fold 统计的评分分布；
- `augmentation_preview.jpg`：canonical crop 及其三个随机增强版本。

五折在患者级别划分。同一患者的颈椎、腰椎图像始终位于同一个 fold。

## Crop 方向约定

Stage 1 预测的两个相邻椎体中心构成 crop 的 posterior 边界。crop 从该边界朝两个
VU 评分角点方向扩展一个椎体中心间距，再变换为 `256×256`。因此所有 crop 都具有
一致的解剖方向：

```text
left = anterior, right = posterior
top  = superior, bottom = inferior
```

禁止水平或垂直翻转，因为翻转会改变上述语义，并可能交换 up/down 两个端点。

## 在线训练增强

`ZhongriVUDataset(..., augment=True)` 每次读取样本时在线生成新的增强结果，不会把
图像复制成一个数量有限的离线增强数据集。

- crop 旋转：±3°；
- crop 两轴平移：各 ±4%；
- 视野缩放：0.90–1.10；
- gamma 和对比度：±10%；
- 归一化强度空间中的亮度：±0.03；
- 轻度灰度高斯噪声：概率 25%；
- 轻度 `3×3` 高斯模糊：概率 15%。

验证和测试 Dataset 必须设置 `augment=False`。明确禁用翻转、MixUp、CutMix、
Random Erasing、弹性形变，以及把缺失标签伪造为 0 分的做法。

对于评分分布不平衡，`build_balanced_sampler(train_samples, power=0.5)` 提供平方根
逆频率采样。它分别计算 up/down 权重后取平均，不会把一个 VU 合并成一个人为组合类别。
设置 `power=0` 可恢复自然分布采样，作为对照基线。

## 训练 EfficientNet-B0 序数评分模型

Stage 2 接收一个经过在线增强的 `256×256` VU，并输出两个单调序数预测。每个输出
包含三个累计判断：`score >= 1`、`score >= 2` 和 `score >= 3`。验证集使用一个
患者级 fold，且不进行数据增强。

训练一个 fold：

```bash
cd /home/lsw/lv/mSASSS_pipeline
bash src/stage2/train_efficientnet.sh --fold 0
```

启用 up/down 局部 ROI：

```bash
bash src/stage2/train_efficientnet.sh --fold 0 --use-roi
```

默认 ROI 配置为 crop 像素空间中的 `64×64` 窗口、`5×5` ROIAlign 输出和 128 维局部
特征。可以分别通过 `--local-roi-size`、`--roi-output-size` 和 `--local-dim` 调整。

依次训练全部五个 fold：

```bash
bash src/stage2/train_all_folds.sh
```

依次训练全部五个 ROI fold：

```bash
bash src/stage2/train_all_folds.sh --use-roi
```

默认配置为 ImageNet 初始化、训练 100 个 epoch、batch size 16、前 5 个 epoch 冻结
backbone、使用平方根评分均衡采样，并在验证集 mean MAE 连续 20 个 epoch 没有改善时
提前停止。输出保存在 `src/stage2/outputs/efficientnet/fold_N/`。

如果需要从无数据泄漏的 Stage 1 EfficientNet checkpoint 初始化：

```bash
bash src/stage2/train_efficientnet.sh \
  --fold 0 \
  --stage1-weights /path/to/stage1/weights/best.pt
```

这里只加载 Stage 1 的 `backbone`，丢弃其 heatmap、FPN 和分类头。严格交叉验证中，
不能使用训练时见过当前 Stage 2 验证患者的 Stage 1 checkpoint。
