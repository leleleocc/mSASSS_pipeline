# Stage 1 EfficientNet-B0-FPN：数据增强、模型架构与训练策略

本文档描述当前 Stage 1 EfficientNet 实现。模型目标是从一张完整的颈椎或腰椎
侧位 X 光片中预测固定顺序的 19 个关键点，并辅助判断图像属于颈椎还是腰椎。

## 1. 任务和数据契约

训练数据与 Stage 1 YOLO 共用同一份 YOLO-Pose 数据集：

```text
src/stage1/data/spine_keypoints_19pt/
├── data.yaml
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

每张图必须恰好有一行标注和 19 个可见关键点：

```text
class cx cy width height x1 y1 v1 ... x19 y19 v19
```

EfficientNet 数据集读取器使用类别和 19 点，不直接使用标签中的检测框。

| 划分 | 图像数 |
|---|---:|
| train | 4,539 |
| val | 599 |
| test | 600 |
| 总计 | 5,738 |

类别为 `cervical_spine` 和 `lumbar_spine`。关键点为七个椎体中心加六组 VU
角点，顺序与 Stage 1 YOLO 完全一致。

## 2. 输入预处理

默认输入尺寸：

```text
1024 × 768（height × width）
```

### 2.1 Letterbox

所有图像首先进行等比例 letterbox：

1. 计算 `scale = min(input_width/original_width, input_height/original_height)`；
2. 使用相同比例缩放横纵坐标；
3. 将缩放后的图像居中放入黑色画布；
4. 对所有关键点应用相同的 scale 和 padding；
5. 不进行非等比拉伸。

返回的 `resize_scale`、`pad` 和原图尺寸会被保留，验证时用于将预测点精确映射回
原图像素坐标。

### 2.2 颜色和归一化

OpenCV 读取到的 BGR 图像会转换为 RGB，然后使用 ImageNet 统计量归一化：

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

这样与 torchvision EfficientNet-B0 的 ImageNet 权重保持一致。

目标关键点最终归一化到 `[0,1] × [0,1]`，归一化分母为
`[input_width-1, input_height-1]`。

## 3. 数据增强策略

增强仅用于训练集。验证和测试只有 letterbox、RGB 转换和 ImageNet 归一化。

### 3.1 随机仿射

默认参数：

| 参数 | 默认值 | 范围 |
|---|---:|---|
| rotation | 5° | `[-5°, +5°]` |
| translation | 0.03 | 横纵各 `[-3%, +3%]` |
| scale | 0.10 | `[0.90, 1.10]` |

变换在完成 letterbox 后执行。每次最多尝试 12 组随机参数，只有当全部 19 个点都
落在图像内部安全边界时才接受：

```text
2 ≤ x ≤ width - 3
2 ≤ y ≤ height - 3
```

若 12 次均无法保证全部关键点有效，则返回未做仿射的图像和坐标。这一策略避免
边缘关键点被裁掉后仍以不可见点参与强监督。

当前仿射没有 shear、透视、弹性形变，也不允许水平或垂直翻转。

### 3.2 X 光强度增强

仿射后进行保守的 X 光强度增强：

| 增强 | 默认范围或概率 |
|---|---|
| gamma | `[0.90, 1.10]` |
| contrast | `[0.90, 1.10]` |
| brightness | `[-0.03, +0.03]` |
| Gaussian noise | 概率 0.25 |
| noise sigma | 0.008，像素已归一化到 `[0,1]` |

增强后裁剪到 `[0,1]`。Letterbox 产生的纯黑 padding 会通过 mask 恢复为 0，
防止亮度和噪声增强把 padding 变成可被模型利用的伪特征。

Stage 1 EfficientNet 当前没有随机模糊、CLAHE、随机擦除、MixUp、CutMix、Mosaic
或 Copy-Paste。

## 4. 模型架构

### 4.1 总体结构

```text
RGB image [B,3,1024,768]
        │
        ▼
EfficientNet-B0 feature backbone
        │
        ├─ stride 4  / 24 channels
        ├─ stride 8  / 40 channels
        ├─ stride 16 / 112 channels
        └─ stride 32 / 1280 channels
        │
        ├─────────────► global pool → Linear(1280,2) → C/L class logits
        │
        ▼
64-channel top-down FPN
        │
        ▼
stride-4 heatmap head
        │
        ▼
19 heatmaps → differentiable soft-argmax → [B,19,2]
```

### 4.2 Backbone

使用 torchvision `efficientnet_b0().features`。默认加载
`EfficientNet_B0_Weights.IMAGENET1K_V1`。特征抽取位置为 backbone stage
索引 `2、3、5、8`，对应通道数 `24、40、112、1280`。

### 4.3 FPN

四个尺度先通过 lateral block 统一成 64 通道。每个 block 为：

```text
Conv2d(bias=False) → GroupNorm(8 groups) → SiLU
```

之后从 stride 32 开始逐级双线性上采样并与 lateral 特征相加：

```text
p32 → p16 → p8 → p4
```

每一级融合后使用 3×3 `Conv + GroupNorm + SiLU` 平滑。GroupNorm 不依赖 batch
统计量，适合高分辨率输入下较小的 batch。

### 4.4 Heatmap head 与 soft-argmax

最终 head 为：

```text
3×3 Conv/GN/SiLU → 1×1 Conv → 19-channel heatmaps
```

每个 heatmap 在空间维做 softmax，形成概率分布；再计算概率分布对规则网格
`x,y∈[0,1]` 的期望，得到可微坐标。与硬 argmax 相比，坐标损失能够直接反向传播
到整张 heatmap。

### 4.5 辅助分类头

最深层 1280 通道特征经过全局平均池化，再由 `Linear(1280,2)` 输出颈椎/腰椎
分类 logits。该分类不是最终任务，但可帮助 backbone 学习视野和解剖区域差异。

## 5. 损失函数

总损失为：

```text
L_total = L_heatmap
        + coordinate_gain × L_coordinate
        + structure_gain  × L_structure
        + class_gain      × L_class
```

默认 gain：

| 项 | gain |
|---|---:|
| heatmap | 1.0 |
| coordinate | 10.0 |
| structure | 2.0 |
| auxiliary class | 0.1 |

### 5.1 Heatmap loss

根据归一化目标坐标在 stride-4 heatmap 上生成二维高斯分布，默认
`sigma=2.0` 个 heatmap 像素。每个目标 heatmap 归一化为概率分布，预测 heatmap
使用 log-softmax，二者计算 KL divergence。

### 5.2 Coordinate loss

soft-argmax 坐标与目标归一化坐标之间使用 Smooth L1：

```text
beta = 0.01
```

### 5.3 Spine structure loss

结构损失与 YOLO 版本使用相同拓扑：

```text
L_structure = 0.5 × L_adjacent
            + 0.3 × L_pair
            + 0.2 × L_order
```

- `adjacent`：匹配 B 链、P1 链、P2 链的相邻向量；
- `pair`：匹配六组 `VxP1 → VxP2`；
- `order`：保持每个病例标注中的上下方向，margin 为 0.002。

向量目标来自每个病例自身，不是人口平均形状。

### 5.4 Auxiliary classification loss

颈椎/腰椎分类使用标准 cross entropy，并以 0.1 的较小权重加入总损失。

## 6. 默认训练参数

下表以当前 `train_efficientnet.py` 为准：

| 参数 | 默认值 |
|---|---:|
| epochs | 100 |
| input shape | `1024 × 768`，H×W |
| batch | 16 |
| workers | 4 |
| device | `0` |
| seed | 42 |
| backbone LR | `1e-4` |
| FPN/head LR | `3e-4` |
| weight decay | `1e-4` |
| warmup epochs | 5 |
| early-stopping patience | 20 |
| Gaussian target sigma | 2.0 |
| coordinate gain | 10.0 |
| structure gain | 2.0 |
| class gain | 0.1 |
| pretrained | ImageNet，默认开启 |
| AMP | 默认开启 |
| gradient clipping | max norm 5.0 |

## 7. 优化与调度策略

### 7.1 参数组

优化器为 AdamW，分为两个学习率组：

1. EfficientNet backbone：`1e-4`；
2. FPN、heatmap head、辅助分类头：`3e-4`。

所有层从第一轮开始训练，没有冻结 backbone 的阶段。

### 7.2 BatchNorm

每轮调用 `model.train()` 后，所有 BatchNorm 模块单独切换为 eval 模式，因此：

- running mean/variance 保持 ImageNet 预训练状态；
- BatchNorm affine 参数仍可接收梯度；
- FPN 使用 GroupNorm，不受该设置影响。

这是针对高分辨率、小 batch 训练的稳定性策略。

### 7.3 学习率调度

使用 `LambdaLR`：

1. 前 5 个 epoch 线性 warmup；
2. 后续采用 cosine decay；
3. 最低倍率为基础学习率的 5%。

warmup 倍率实现为：

```text
0.1 + 0.9 × (epoch + 1) / warmup_epochs
```

### 7.4 AMP、梯度和复现

- CUDA 下默认启用 AMP；
- 反向传播后执行 `clip_grad_norm_(max_norm=5.0)`；
- Python、NumPy、PyTorch 和各 DataLoader worker 均设置随机种子；
- checkpoint 保存模型、优化器、scheduler、GradScaler、epoch 和早停状态；
- 支持从 `last.pt` 完整恢复。

## 8. 多 GPU 策略

`train_efficientnet.sh` 默认：

```text
STAGE1_NPROC_PER_NODE=8
```

大于 1 时使用 `torch.distributed.run` 和 NCCL DDP。训练集使用
`DistributedSampler`，每轮调用 `set_epoch(epoch)` 更新 shuffle。

验证只在 rank 0 上执行完整验证集。训练 loss 日志来自 rank 0 本地 shard，当前没有
跨 rank all-reduce；它适合观察趋势，但不是整个分布式 epoch 的严格全局均值。

单 GPU 运行：

```bash
STAGE1_NPROC_PER_NODE=1 bash src/stage1/train_efficientnet.sh --device 0
```

## 9. 验证、模型选择和早停

预测坐标先根据保存的 scale 和 padding 映射回原图。主要指标包括：

- mean、median、p95 像素误差；
- normalized MRE：欧氏误差除以原图对角线；
- PCK@0.5%、PCK@1%、PCK@2% image diagonal；
- 颈椎/腰椎分类准确率。

用于最佳模型选择的是：

```text
mean_error_image_diag_pct
```

连续 20 个 epoch 没有降低时提前停止。`best.pt` 去除优化器、scheduler 和 scaler
状态，用于推理；`last.pt` 保留完整状态，用于恢复训练。

## 10. 运行命令

默认多 GPU 训练：

```bash
bash src/stage1/train_efficientnet.sh
```

单 GPU 示例：

```bash
STAGE1_NPROC_PER_NODE=1 bash src/stage1/train_efficientnet.sh \
  --device 0 \
  --batch 8 \
  --name efficientnet_b0_19pt_single_gpu
```

关闭 ImageNet 和 AMP：

```bash
STAGE1_NPROC_PER_NODE=1 bash src/stage1/train_efficientnet.sh \
  --device 0 \
  --no-pretrained \
  --no-amp
```

独立评估：

```bash
bash src/stage1/evaluate_efficientnet.sh \
  src/stage1/outputs/efficientnet/efficientnet_b0_19pt/weights/best.pt \
  --split val \
  --name efficientnet_val
```

## 11. 输出产物

默认训练目录：

```text
src/stage1/outputs/efficientnet/efficientnet_b0_19pt/
├── args.json
├── results.csv
├── results.png
└── weights/
    ├── best.pt
    └── last.pt
```

独立评估会另外生成逐病例、逐关键点、分来源、分类别的统计与可视化。

## 12. 实验约束

1. 训练和评估必须使用相同输入尺寸及 H×W 顺序；
2. 不要启用任何翻转，除非同时重新定义关键点索引和上下/左右语义；
3. 比较 YOLO 与 EfficientNet 时统一使用原图 normalized MRE/PCK；
4. 保留患者/病例级拆分，避免同一病例跨 train/val/test；
5. 修改 heatmap stride 或输入分辨率时，应重新评估 `sigma=2.0` 是否合理；
6. 如果 batch 足够大并希望更新 BatchNorm running statistics，需要把该策略作为独立实验，不能与当前基线直接混用。

## 13. 实现索引

- [模型、Dataset、增强、损失和内部验证](../efficientnet.py)
- [训练入口](../train_efficientnet.py)
- [独立评估入口](../evaluate_efficientnet.py)
- [数据集关键点定义](../data/spine_keypoints_19pt/keypoint_schema.json)
- [数据集统计](../data/spine_keypoints_19pt/dataset_summary.json)
