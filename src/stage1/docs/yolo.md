# Stage 1 YOLO26s-Pose：数据增强、模型架构与训练策略

本文档描述当前代码中的 Stage 1 YOLO 实现。代码事实以
`train_yolo_pose.py`、`yolo_strategy.py` 和当前安装的 Ultralytics 行为为准。

## 1. 任务定义

Stage 1 接收一张颈椎或腰椎侧位 X 光片，同时完成：

1. 整段目标的单实例检测；
2. 颈椎/腰椎二分类；
3. 固定顺序的 19 个关键点定位。

每张图只允许一个 pose 实例。YOLO 标签由类别、检测框和 19 组
`(x, y, visibility)` 组成，坐标均相对图像宽高归一化：

```text
class cx cy width height x1 y1 v1 ... x19 y19 v19
```

数据集类别为：

| 类别 ID | 类别 |
|---:|---|
| 0 | `cervical_spine` |
| 1 | `lumbar_spine` |

通用关键点顺序为：

```text
B1, B2, B3, B4, B5, B6, B7,
V1P1, V1P2, V2P1, V2P2, ... , V6P1, V6P2
```

- `B1..B7`：从上到下的七个椎体中心；
- `VxP1`：该 VU 上方椎体的图像左下角；
- `VxP2`：该 VU 下方椎体的图像左上角。

颈椎映射为 C2、C3、C4、C5、C6、C7、T1；腰椎映射为
T12、L1、L2、L3、L4、L5、S1。

当前生成数据共 5,738 张：训练 4,539 张、验证 599 张、测试 600 张。
数据来自 Zhongri、CSXA 和 BUU-LSPINE，拆分和去重在 Stage 1 数据准备阶段完成。

## 2. 输入尺寸和坐标变换

默认网络输入固定为：

```text
height = 1024
width  = 768
```

两个维度都必须不小于 64，并且可以被 32 整除。命令行 `--imgsz` 接受：

- 一个整数：生成正方形输入；
- 两个整数：按 `height width` 解释。

训练入口仍向 Ultralytics 传入 `imgsz=max(height, width)`，但自定义
`FixedShapePoseDataset` 会覆盖最终变换，确保张量严格为指定的矩形尺寸。
图像采用统一比例的缩放、平移、裁切或填充，不进行横纵方向不一致的拉伸。

验证阶段使用 `LetterBox(new_shape=(height, width), scaleup=False)`：

- 保持原始长宽比；
- 只缩小大图，不主动放大小图；
- 用 padding 补齐到固定尺寸；
- 关键点和检测框同步映射。

## 3. 数据增强策略

### 3.1 几何增强

训练集的核心几何增强由 `RandomPerspective` 完成，当前默认值如下：

| 参数 | 默认值 | 实际范围或含义 |
|---|---:|---|
| `degrees` | 5.0 | 随机旋转约 `[-5°, +5°]` |
| `translate` | 0.03 | 横纵方向最多平移输入尺寸的 3% |
| `scale` | 0.10 | 等比例缩放约 `[0.90, 1.10]` |
| `shear` | 0.0 | 禁用剪切 |
| `perspective` | 0.0 | 禁用透视畸变 |

这是偏保守的医学影像增强。小幅旋转、平移和缩放用于模拟摆位差异及
Stage 1 对视野变化的鲁棒性，同时避免生成不可信的脊柱形态。

### 3.2 强度和颜色增强

YOLO 训练显式设置：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `hsv_h` | 0.0 | 不改变色相 |
| `hsv_s` | 0.0 | 不改变饱和度 |
| `hsv_v` | 0.10 | 允许轻微亮度变化 |

`FixedShapePoseDataset` 还构造了 Ultralytics 的 `Albumentations` 包装器。
当前调用没有在项目代码中提供自定义 transform 列表，因此其具体默认变换
取决于实际安装的 Ultralytics/Albumentations 版本。复现实验时应保存包版本；
若要求增强策略完全固定，应在项目中显式传入变换列表。

### 3.3 明确禁用的增强

以下增强被显式设为 0：

| 增强 | 状态 | 原因 |
|---|---|---|
| 水平翻转 `fliplr` | 禁用 | 会交换图像左右/前后语义 |
| 垂直翻转 `flipud` | 禁用 | 会交换上下椎体及 VU 两端 |
| Mosaic | 禁用 | 一张图必须对应一套连续脊柱结构 |
| MixUp | 禁用 | 会混合两套互不对应的关键点 |
| CutMix | 禁用 | 会破坏椎体链的连续性 |
| Copy-Paste | 禁用 | 会生成解剖学上不合理的重复结构 |
| shear | 禁用 | 避免非生理性形变 |
| perspective | 禁用 | 避免强投影畸变 |

即使数据 YAML 中存在 `flip_idx`，当前训练也不会执行翻转。

### 3.4 训练和验证差异

```text
训练：RandomPerspective → Albumentations → RandomHSV → Format
验证：LetterBox → Format
```

验证、测试和正式评估不使用随机增强。

## 4. 模型架构

### 4.1 基础模型

默认权重为：

```text
yolo26s-pose.pt
```

实际 backbone、neck 和内部通道数由该 Ultralytics checkpoint/模型配置决定，
不在本仓库中重新声明。本仓库扩展的是 Pose 模型类别、损失、数据管线和验证器：

```text
YOLO26s backbone
    ↓
多尺度 neck
    ↓
Pose26 head
    ├─ box / object assignment
    ├─ cervical vs lumbar class
    ├─ 19 × (x, y) keypoint regression
    └─ 19 keypoint visibility/objectness
```

`SpinePoseModel` 继承 Ultralytics `PoseModel`，并根据模型 head 选择：

- 普通 Pose26：`SpinePoseLoss`；
- end-to-end Pose26：`E2ELoss(self, SpinePoseLoss)`；
- 非 Pose26 兼容路径：回退到 `v8PoseLoss`。

因此，模型文件和 Ultralytics 版本是架构的一部分。当前环境安装脚本没有固定
Ultralytics 精确版本，正式实验应额外记录 `pip freeze`、checkpoint 哈希和模型摘要。

### 4.2 自定义脊柱结构损失

YOLO 原生 pose 损失之外，项目增加了 19 点结构损失。所有坐标先相对匹配到的
目标框进行归一化，使结构项主要描述病例自身的相对几何，而不是绝对像素大小。

结构损失包含三部分：

1. `adjacent`：匹配三个从上到下的关键点链的相邻向量；
2. `pairs`：匹配六组 `VxP1 → VxP2` 向量；
3. `order`：惩罚预测点违反标注所定义的上下顺序。

三个链为：

```text
B1 → B2 → ... → B7
V1P1 → V2P1 → ... → V6P1
V1P2 → V2P2 → ... → V6P2
```

内部组合为：

```text
L_structure = 0.5 × L_adjacent
            + 0.3 × L_pairs
            + 0.2 × L_order
```

`L_adjacent` 和 `L_pairs` 使用 Smooth L1，`beta=0.01`；`L_order` 使用
`ReLU(0.002 - directed_vertical_step)`。最终附加项为：

```text
L_added = structure_gain × L_structure
```

默认 `structure_gain=1.0`。该损失匹配每个病例自己的目标向量，并不要求所有脊柱
接近某个平均模板，因此不会主动压制真实的异常曲度或结构变异。

### 4.3 原生 YOLO 损失

模型仍保留 Ultralytics Pose26 的原生检测、分类、分布、关键点位置和关键点可见性
相关损失。各原生项的精确定义和默认 gain 由安装的 Ultralytics 版本决定；本仓库
仅显式增加 `structure_loss`，没有覆盖其余 gain。

## 5. 默认训练参数

下表以当前 `train_yolo_pose.py` 为准：

| 参数 | 默认值 |
|---|---:|
| model | `yolo26s-pose.pt` |
| epochs | 100 |
| input shape | `1024 × 768`，顺序为 H×W |
| global batch | 64 |
| device | `0,1,2,3,4,5,6,7` |
| workers | 4 |
| seed | 42 |
| deterministic | `True` |
| optimizer | AdamW |
| initial LR `lr0` | `1e-3` |
| weight decay | `5e-4` |
| early-stopping patience | 20 |
| structure gain | 1.0 |
| pretrained | 开启 |
| AMP | 开启 |
| validation | 每轮开启 |
| plots | 开启 |

未在入口中显式传入的优化器细节、warmup、最终学习率比例等参数继承
Ultralytics 默认配置。它们不是本项目稳定 API，升级 Ultralytics 后必须重新核对。

## 6. 训练策略

### 6.1 初始化

默认从 `yolo26s-pose.pt` 预训练权重开始。`--no-pretrained` 会向训练器传入
`pretrained=False`，但模型文件本身如何解析仍由 Ultralytics 负责。

### 6.2 优化和数值稳定性

- 优化器：AdamW；
- 自动混合精度：默认开启，可用 `--no-amp` 禁用；
- 确定性模式：开启；
- 每轮执行验证并生成 Ultralytics 图表；
- 自定义损失在有正样本匹配时计算，无匹配时保持可微的零值。

### 6.3 多 GPU

默认 `--device` 指定 8 张 GPU，由 Ultralytics 自行管理多 GPU 启动和训练。
批大小 64 是全局还是每卡语义应以当前 Ultralytics 版本的实际日志为准。
单卡运行时显式传入例如：

```bash
bash src/stage1/train_yolo_pose.sh --device 0 --batch 8
```

### 6.4 Early stopping 和最佳模型

自定义 `NormalizedMREPoseValidator` 在原图坐标中计算：

```text
normalized error = Euclidean point error / original image diagonal
```

聚合指标包括：

- normalized MRE；
- PCK@1% image diagonal；
- Ultralytics 原生 box/pose 指标。

若某图没有检测结果，该图 19 个点的归一化误差均按 1.0 计，避免漏检样本被排除。
用于 checkpoint 选择的 fitness 为：

```text
fitness = 100 - normalized_MRE_percent
```

因此最佳权重优先对应最低的验证集 normalized MRE，而不是最高 mAP。

## 7. 运行命令

默认训练：

```bash
bash src/stage1/train_yolo_pose.sh
```

单 GPU、小 batch 示例：

```bash
bash src/stage1/train_yolo_pose.sh \
  --device 0 \
  --batch 8 \
  --name yolo26s_pose_19pt_single_gpu
```

调整结构损失和几何增强：

```bash
bash src/stage1/train_yolo_pose.sh \
  --structure-gain 0.5 \
  --degrees 3 \
  --translate 0.02 \
  --scale 0.08
```

独立评估：

```bash
bash src/stage1/evaluate_yolo_pose.sh \
  src/stage1/outputs/yolo/yolo26s_pose_19pt/weights/best.pt \
  --split val \
  --name yolo_val
```

独立评估默认置信度阈值为 `0.001`、每图最多保留一个实例，并同时生成统一的
原图像素误差、归一化误差、PCK、分类准确率、逐点统计和最差病例可视化。

## 8. 训练产物

默认输出目录：

```text
src/stage1/outputs/yolo/yolo26s_pose_19pt/
```

主要产物由 Ultralytics 生成，包括：

- `weights/best.pt`；
- `weights/last.pt`；
- 训练结果 CSV；
- loss 和验证指标曲线；
- 验证预测可视化。

独立评估另写入 `src/stage1/outputs/evaluation/yolo/`。

## 9. 复现与安全约束

1. 固定数据集 `manifest.json`、数据 YAML 和权重文件哈希；
2. 固定 Python、PyTorch、Ultralytics 和 Albumentations 版本；
3. 不要开启水平/垂直翻转、Mosaic 或混合样本增强；
4. 训练、验证、测试必须保持当前患者/病例级拆分；
5. 比较模型时统一使用原图坐标 normalized MRE 和相同的缺失检测惩罚；
6. 修改输入尺寸时必须保持 H×W 顺序、两维均可被 32 整除，并同步用于训练和评估。

## 10. 实现索引

- [训练入口](../train_yolo_pose.py)
- [固定尺寸数据集、结构损失和验证器](../yolo_strategy.py)
- [独立评估入口](../evaluate_yolo_pose.py)
- [数据集配置](../data/spine_keypoints_19pt/data.yaml)
