# Stage 2 EfficientNet：up/down 局部 ROI 架构与训练策略

本文档定义 Stage 2 的目标方案：Stage 1 保持现有 19 点定位任务不变，Stage 2 使用
Stage 1 输出的椎体中心和 VU 端点，构造方向统一的 VU 图像，并显式提取 up/down
两个端点的局部 ROI 特征，最终分别预测两个 mSASSS 等级 `0..3`。

本文会区分两种状态：

- **当前基线**：仓库已经实现，仅使用整块 VU 的全局 EfficientNet 特征；
- **当前 ROI 模型**：已实现 stride-8 双 ROIAlign、全局/局部融合和双序数头；
- **后续增强项**：小型 FPN、geometry MLP 和 Stage 1 OOF predicted ROI 训练。

Stage 2 不再考虑 YOLO。它是一个已知端点位置后的细粒度序数分类问题，而不是第二次
关键点检测。

## 1. 任务定义

### 1.1 Stage 1 输出

每张颈椎或腰椎侧位片由 Stage 1 输出 19 个点：

```text
B1 ... B7
V1P1, V1P2, ... , V6P1, V6P2
```

其中：

- `B_i` 和 `B_(i+1)` 是第 `i` 个 VU 的相邻椎体中心；
- `V_iP1` 是空间上方的评分角点，即 Stage 2 的 `up point`；
- `V_iP2` 是空间下方的评分角点，即 Stage 2 的 `down point`。

Stage 1 数据准备代码按 `B_i → B_(i+1)` 方向上的投影对 P1/P2 排序，因此 Stage 2
不应根据原图的 x/y 大小重新猜测 up/down。接口必须保留点的语义顺序。

### 1.2 Stage 2 输出

每个 VU 输出两个独立评分：

```text
up_score   ∈ {0, 1, 2, 3}
down_score ∈ {0, 1, 2, 3}
```

二者不是一个 16 类标签，也不存在 `up_score <= down_score` 或相反的约束。缺失评分
不能转成 0。

### 1.3 为什么仍需要全局 VU

评分角点附近包含骨赘、侵蚀和硬化等局部征象，但桥接需要观察相邻椎体之间的连续性。
因此只裁两个小 patch 会损失椎间隙和跨端点关系。推荐模型同时保留：

- 全局 VU：椎体边缘、椎间隙和桥接关系；
- up ROI：上端评分角附近的细节；
- down ROI：下端评分角附近的细节。

两个评分头都读取 up 和 down ROI，而不是各自只看一个端点。

## 2. 数据集与划分

Stage 2 的评分监督只来自 Zhongri：

| 项目 | 数量 |
|---|---:|
| 患者 | 58 |
| 原始侧位片 | 116 |
| 每张图 VU | 6 |
| VU 样本 | 696 |
| up/down 端点评分 | 1,392 |

级别映射为：

```text
颈椎：C2-C3, C3-C4, C4-C5, C5-C6, C6-C7, C7-T1
腰椎：T12-L1, L1-L2, L2-L3, L3-L4, L4-L5, L5-S1
```

当前标签分布：

| 分数 | up | down |
|---:|---:|---:|
| 0 | 37 | 71 |
| 1 | 294 | 351 |
| 2 | 266 | 178 |
| 3 | 99 | 96 |

五折必须按患者划分。一位患者的颈椎、腰椎和全部 VU 永远处于同一个 fold。

CSXA 和 BUU-LSPINE 没有 mSASSS 分数，不能加入 Stage 2 的评分监督；它们只可用于
Stage 1 定位训练。

## 3. 从 19 点构造一个 VU 样本

### 3.1 Canonical VU 坐标系

相邻椎体中心定义 VU 纵轴：

```text
unit_y = normalize(B_(i+1) - B_i)
```

VU 两端点的中点用于判断椎体中心连线的 anterior 一侧。最终 crop 固定为：

```text
left   = anterior
right  = posterior
top    = superior
bottom = inferior
```

源图四边形通过透视变换得到 `256×256` VU crop。当前代码以相邻中心距离为边长，
中心连线是 crop 的 posterior 边界，crop 向两个评分点所在的 anterior 方向展开。

### 3.2 把 up/down 点映射到 crop

构造 VU crop 时会得到从原图到 crop 的单应矩阵 `H`。同一个 `H` 必须用于转换
`V_iP1` 和 `V_iP2`：

```text
[u, v, w]^T = H · [x, y, 1]^T
x_crop = u / w
y_crop = v / w
```

模型接口使用：

```text
point_xy.shape = [B, 2, 2]
point_xy[:, 0] = up point
point_xy[:, 1] = down point
```

推荐在 Dataset 中同时返回像素坐标和归一化坐标：

```text
point_xy_px   ∈ crop 像素坐标，范围通常为 [0, 255]
point_xy_norm = point_xy_px / 255，范围通常为 [0, 1]
```

点在 crop 外时不要通过移动 ROI 中心来伪造位置。保留真实中心，让 ROIAlign 对越界部分
补零，同时输出 `point_valid`/`geometry_valid` 质量标记。

### 3.3 几何增强必须同步变换点

训练时只要改变了 crop 的旋转、平移或视野大小，就必须使用同一变换重新计算
`point_xy`。禁止只增强图像而保留增强前的点坐标，否则局部 ROI 会取错位置。

## 4. 推荐的 up/down ROI 定义

默认在 canonical VU 坐标中，以两个端点为中心，各建立一个 `64×64` 局部窗口：

```text
up_box   = box(center=up point,   size=64×64)
down_box = box(center=down point, size=64×64)
```

`64/256 = 25%` 的视野通常足以包含角点附近骨皮质和小骨赘，同时不会退化成另一个
全局 crop。推荐消融 `48、64、80`，而不是一开始引入多尺度三分支。

局部窗口只用于特征采样，不另存两份图像，也不分别运行两次 EfficientNet。这样 up、
down 和全局分支共享同一套图像增强和 backbone 特征，显存与参数量也更可控。

## 5. 模型架构

### 5.1 总览

```text
VU image [B,3,256,256]
        │
        ▼
EfficientNet-B0 shared backbone
        │
        ├── C8 [B,40,32,32]
        │           │
        │      up/down boxes from point_xy
        │           │
        │      ROIAlign(5×5, spatial_scale=1/8)
        │           ├── up local   128-D
        │           └── down local 128-D
        │
        └── C32 [B,1280,8,8] ─> global pooling ─> global 256-D

[global, up local, down local]
        │
        ├── up-aware fusion   ─> monotonic ordinal head ─> up logits [B,3]
        └── down-aware fusion ─> monotonic ordinal head ─> down logits [B,3]
```

以上是当前 `--use-roi` 实现。通道数对应 torchvision EfficientNet-B0；代码从明确的
feature stage 取 `C8`，最终 `C32` 继续用于全局分支。EfficientNet 对每个 VU 只运行一次。

### 5.2 可选的小型 FPN

当前第一版直接在 C8 上执行 ROIAlign。后续若消融实验表明局部特征语义不足，可加入
stride-16 特征形成小型 FPN：

```text
P8 = Conv3x3(
       Conv1x1(C8, 128)
       + Upsample(Conv1x1(C16, 128), size=C8)
     )
```

该 FPN 尚未实现。新增时推荐 GroupNorm，而不是为小 batch 新增 BatchNorm。`P8` 保持
`32×32` 空间分辨率；输入上的 `64×64` ROI 对应约 `8×8` 个 P8 特征位置。

### 5.3 局部 ROI 编码

把每个 box 转为 torchvision ROIAlign 格式：

```text
[batch_index, x1, y1, x2, y2]
```

当前默认参数：

| 参数 | 默认值 |
|---|---:|
| input ROI size | `64×64` |
| feature stride | 8 |
| ROIAlign output | `5×5` |
| sampling ratio | 2 |
| aligned | `True` |

ROIAlign 后执行：

```text
AdaptiveAvgPool2d(1)
→ Linear(40,128)
→ LayerNorm
→ SiLU
→ Dropout(0.20)
```

up/down 使用完全共享的 local encoder，避免小数据下两个分支学习到不必要的参数差异。

### 5.4 全局 VU 编码

最终 stride-32 特征执行：

```text
AdaptiveAvgPool2d(1)
→ Linear(1280,256)
→ LayerNorm
→ SiLU
→ Dropout(0.30)
```

这一路负责椎间隙、相邻椎体边缘和桥接等跨区域信息。

### 5.5 可选的几何编码

当前第一版不使用 geometry MLP。若 ROI 模型稳定后仍需要显式位置关系，可加入：

```text
q_up   = [x_up,   y_up,   x_down, y_down,  dx,  dy, distance]
q_down = [x_down, y_down, x_up,   y_up,   -dx, -dy, distance]
```

其中 `dx=x_down-x_up`、`dy=y_down-y_up`。所有值在 canonical crop 中归一化，再经共享的
两层 MLP 输出 32 维。它属于后续消融项，不能替代局部影像特征。

`view_id` 和 `level_index` 可分别使用 8 维 embedding，但第一版建议默认关闭，并作为
消融项测试。样本量很小，过早加入级别信息可能让模型记忆等级先验而忽略影像。

### 5.6 up/down-aware 融合

两个头都看两个端点，但输入顺序强调当前要评分的端点：

```text
z_up   = Fusion([global, local_up,   local_down])
z_down = Fusion([global, local_down, local_up])
```

当前 `Fusion` 在 up/down 间共享权重，结构为：

```text
Linear(512,256)
→ LayerNorm
→ SiLU
→ Dropout(0.30)
→ Linear(256,256)
→ SiLU
```

这里的 `512 = 256 + 128 + 128`。将来加入 32 维 geometry 后，输入维度才变为 544。

这种设计允许桥接信息从另一端影响当前端点评分，又不会把两个评分合成一个标签。

### 5.7 单调序数头

每个 `0..3` 分数编码为三个累计判断：

```text
score >= 1, score >= 2, score >= 3
```

| 原始评分 | ordinal target |
|---:|---|
| 0 | `[0,0,0]` |
| 1 | `[1,0,0]` |
| 2 | `[1,1,0]` |
| 3 | `[1,1,1]` |

每个 endpoint 学习一个 latent score 和三个有序阈值：

```text
t1 = threshold_start
t2 = t1 + softplus(gap1)
t3 = t2 + softplus(gap2)
logit_k = latent_score - tk
```

因此天然保证：

```text
P(score >= 1) >= P(score >= 2) >= P(score >= 3)
```

这里的“有序”只指同一个评分的三个阈值，不表示 up 和 down 分数之间存在大小关系。

## 6. 损失函数

### 6.1 主损失

第一版只需要评分监督：

```text
L_up    = mean BCE(up logits, up ordinal target)
L_down  = mean BCE(down logits, down ordinal target)
L_score = (L_up + L_down) / 2
```

up/down 默认等权。若后续发现某个 endpoint 的标注噪声或有效样本量不同，再基于训练集
统计设权重，不要根据验证结果手调。

### 6.2 不加入 Stage 1 的结构损失

Stage 2 第一版不重新预测关键点，因此不使用 Stage 1 的 `L_adjacent`、`L_pair`、
`L_order`。这些损失约束的是点几何结构，不应加到纯评分输出上。

如果以后增加可学习的端点残差：

```text
refined_point = stage1_point + bounded_offset
```

才需要额外的点回归损失和偏移正则；这不属于当前第一版。

### 6.3 可选一致性损失

在主模型稳定后，可让同一 VU 的 GT 几何版本和 predicted/noisy 几何版本输出接近：

```text
L_consistency = SmoothL1(sigmoid(logits_a), sigmoid(logits_b))
L_total = L_score + 0.1 × L_consistency
```

它是鲁棒性实验项，默认先设为 0，避免在基线未建立时增加变量。

## 7. 数据增强

### 7.1 当前安全的强度增强

| 增强 | 默认范围或概率 |
|---|---|
| gamma | `[0.90, 1.10]` |
| contrast | `[0.90, 1.10]` |
| brightness | `[-0.03, +0.03]` |
| Gaussian noise | 概率 0.25，sigma 0.008 |
| 3×3 Gaussian blur | 概率 0.15，sigma `[0.1,0.8]` |

所有强度处理在 `[0,1]` 浮点范围内执行。黑色 padding 在增强后恢复为 0，最后按
ImageNet RGB mean/std 归一化。

### 7.2 crop 几何抖动

当前默认参数可保留：

| 参数 | 默认值 | 采样范围 |
|---|---:|---|
| rotation | 3° | `[-3°, +3°]` |
| translation x/y | 0.04 | 各轴 `[-4%, +4%] × center distance` |
| field scale | 0.10 | `[0.90,1.10] × center distance` |

几何抖动后的 `H` 必须同时作用于影像、up point 和 down point。

### 7.3 Stage 1 点误差增强

普通 crop jitter 不能完全模拟 19 个预测点之间的误差。推荐从 leak-safe Stage 1
验证预测中统计以下误差，并按联合分布采样：

- 两个椎体中心的平移与中心距离误差；
- up/down 点相对各自真值的 x/y 误差；
- 两端点误差之间的相关性；
- 颈椎与腰椎的差异。

在尚未得到经验分布前，可使用保守回退值：canonical `256×256` crop 中，up/down
坐标各加二维高斯噪声 `sigma=4 px`，截断到 `±12 px`。这是临时默认值，最终应由
Stage 1 的 OOF 误差分布替换。

训练时可对 local ROI 边长乘 `[0.90,1.10]` 随机尺度，验证和推理固定为 64。

### 7.4 禁止的增强

| 增强 | 原因 |
|---|---|
| 水平翻转 | 交换 anterior/posterior |
| 垂直翻转 | 交换 superior/inferior 和 up/down |
| MixUp/CutMix | 混合两套独立序数标签和解剖结构 |
| Mosaic | 破坏一个输入对应一个 VU 的定义 |
| Random Erasing | 可能直接遮住微小病变或评分角点 |
| Elastic deformation | 可能产生非生理性椎体形态 |

## 8. GT、noisy GT 与 Stage 1 predicted ROI 混合

只用人工点训练、用预测点推理会形成 geometry domain gap。推荐每个训练 VU 随机选择
一种几何来源：

| 来源 | 含义 | 作用 |
|---|---|---|
| GT | 人工中心与端点 | 保留清晰的评分上界 |
| noisy GT | 按 Stage 1 误差扰动人工点 | 连续覆盖定位误差 |
| predicted | Stage 1 OOF 预测点 | 匹配真实部署分布 |

建议课程：

| epoch | GT | noisy GT | predicted | backbone |
|---:|---:|---:|---:|---|
| 1–5 | 70% | 30% | 0% | frozen |
| 6–30 | 40% | 30% | 30% | trainable |
| 31+ | 20% | 30% | 50% | trainable |

如果 predicted ROI 的质量尚未审计，先固定为 `40/30/30`，不要盲目提高比例。

严格 fold `k` 中：

- Stage 2 验证患者不能参与对应 Stage 1 模型的任何监督训练；
- Stage 2 训练图像使用 OOF Stage 1 预测，不能使用见过同一图像的模型预测；
- 验证必须使用 leak-safe Stage 1 预测点模拟完整流水线；
- 同时保留 GT 点验证，作为 scorer 的定位无误差上界。

## 9. 类别不平衡

保留当前平方根逆频率 sampler：

```text
w = 0.5 × [(N / count_up)^power + (N / count_down)^power]
power = 0.5
```

它分别计算 up/down 权重后取平均，不把 `(up,down)` 拼成稀疏的 16 类。`power=0`
用于自然采样基线。

第一版不建议同时启用强采样、class-weighted BCE 和 focal loss。三种校正叠加容易让少量
稀有病例被过度放大。

## 10. 推荐训练参数

目标 ROI 模型可从当前训练参数起步：

| 参数 | 推荐值 |
|---|---:|
| folds | 5，患者级 |
| crop size | 256 |
| local ROI size | 64 |
| ROIAlign output | 5 |
| epochs | 100 |
| batch | 16 |
| workers | 4 |
| seed | 42 |
| backbone LR | `3e-5` |
| ROI/local/fusion/head LR | `3e-4` |
| optimizer | AdamW |
| weight decay | `1e-4` |
| warmup | 5 epochs |
| freeze backbone | 5 epochs |
| LR schedule | warmup + cosine，最低倍率 0.05 |
| early-stopping patience | 20 |
| sampler power | 0.5 |
| global dimension | 256 |
| local dimension | 128 |
| geometry dimension | 当前关闭；后续建议 32 |
| fusion dimension | 256 |
| global/fusion dropout | 0.30 |
| local dropout | 0.20 |
| gradient clipping | max norm 5.0 |
| AMP | 开启 |

冻结 backbone 时，local encoder、fusion 和 ordinal head 从第一轮开始训练。解冻后
backbone 使用小 10 倍的学习率。为保持小数据训练稳定，建议整个训练期间冻结
EfficientNet BatchNorm running statistics；新增模块使用 LayerNorm，未来的 FPN 使用
GroupNorm。

初始化优先级：

```text
resume > leak-safe Stage 1 EfficientNet backbone > ImageNet > random
```

Stage 1 checkpoint 只加载同构 EfficientNet backbone，不加载 heatmap、FPN 或分类头。

## 11. 验证与模型选择

每个 fold 必须运行两套验证输入：

1. **GT-point upper bound**：判断 scorer 本身是否学会评分；
2. **Stage-1-predicted full pipeline**：判断实际部署性能。

每套至少报告：

- up/down 和平均 MAE；
- up/down exact accuracy；
- ±1 accuracy；
- up/down quadratic weighted kappa；
- 两个 `4×4` confusion matrix；
- 按颈椎/腰椎和椎间级别分层的指标。

完整流水线还应报告：

- Stage 1 点误差分位数与 ROI 越界率；
- 单张图 12 个 endpoint 总分 MAE；
- 同一患者颈椎+腰椎总 mSASSS 的 MAE/相关性；
- 定位失败率，失败样本不能静默当作 0 分。

最佳 checkpoint 应优先按 predicted-point 验证集的 `mean_mae` 选择，`val_loss` 作为
tie-break；GT-point 指标用于诊断，不作为最终部署模型的唯一选择依据。

## 12. 最小消融实验

建议至少完成以下四组，使用完全相同的患者 fold：

| 实验 | 全局 | up/down ROI | geometry | predicted/noisy 训练 |
|---|---:|---:|---:|---:|
| A 当前基线 | ✓ |  |  |  |
| B 局部贡献 | ✓ | ✓ |  |  |
| C 完整 point-aware | ✓ | ✓ | ✓ |  |
| D 部署鲁棒模型 | ✓ | ✓ | ✓ | ✓ |

额外验证两个问题：

- 当前端点只看自己的 ROI，还是同时看另一端 ROI；
- ROI size 使用 48、64 还是 80。

如果 B/C 对 GT-point 有提升但 D 的 predicted-point 没提升，问题通常来自 Stage 1 误差
分布或训练/推理 geometry gap，而不是评分头容量不足。

## 13. 失败处理

生成 VU 前检查：

- 19 点数量和语义顺序完整；
- 相邻椎体中心距离大于最小阈值；
- 单应矩阵有限且可逆；
- up/down ROI 与 VU crop 至少有最小交集；
- Stage 1 置信度高于经验证集确定的阈值。

失败时返回 `invalid_geometry` 或低置信度标记，供整片汇总时处理。不要把失败评分写成
0，因为 0 是真实的“无病变”等级。

## 14. “端到端”的边界

部署流水线可以做到一次调用完成：

```text
原始 X-ray
→ Stage 1 预测 19 点
→ 构造 6 个 VU 及 up/down ROI
→ Stage 2 预测 12 个 endpoint 分数
→ 汇总图像/患者 mSASSS
```

这属于业务意义上的端到端推理，Stage 1 不需要改。

当前 OpenCV 透视 crop 不可对 Stage 1 坐标反向传播，因此不是联合可微训练。若以后确实
需要 Stage 2 评分损失反传到 Stage 1，只需把 crop/ROI 生成替换为 PyTorch
`grid_sample`/Spatial Transformer；Stage 1 网络架构本身仍可保持不变。鉴于评分数据只有
116 张，第一版应冻结 Stage 1，先把级联模型做稳。

## 15. 当前代码状态与实现顺序

| 能力 | 当前状态 |
|---|---|
| Zhongri 696 个 VU 与患者级五折 | 已实现 |
| canonical `256×256` VU crop | 已实现 |
| 强度与 crop 几何增强 | 已实现 |
| EfficientNet-B0 全局特征 | 已实现 |
| 双单调序数评分头 | 已实现 |
| up/down 点映射到 crop 坐标 | 已实现 |
| stride-8 C8 双 ROIAlign | 已实现，可用 `--use-roi` 开启 |
| point-aware fusion | 已实现 |
| 小型 FPN | 未实现，后续消融项 |
| geometry MLP | 未实现，后续消融项 |
| Stage 1 OOF predicted ROI 训练/验证 | 未实现 |
| 19 点到 6 VU 的部署推理编排 | 未实现 |

后续推荐实现顺序：

1. 复现全局基线，并用 `--use-roi` 跑相同 fold 的 ROI 对照；
2. 生成 leak-safe Stage 1 OOF 预测并评估完整流水线；
3. 增加 `point_valid`/`geometry_valid` 和定位失败处理；
4. 根据消融结果决定是否增加 FPN 和 geometry MLP；
5. 完成 19 点到 6 个 VU、12 个 endpoint 分数的部署推理编排。

## 16. 运行命令

准备数据并运行全局特征基线：

```bash
bash src/stage2/prepare_data.sh --exist-ok
bash src/stage2/train_efficientnet.sh --fold 0
bash src/stage2/train_all_folds.sh
```

启用当前已实现的 up/down ROI：

```bash
bash src/stage2/train_efficientnet.sh --fold 0 --use-roi
```

ROI 默认参数为 `--local-roi-size 64 --roi-output-size 5 --local-dim 128`。

使用 leak-safe Stage 1 EfficientNet backbone 初始化：

```bash
bash src/stage2/train_efficientnet.sh \
  --fold 0 \
  --stage1-weights /path/to/leak-safe-stage1-best.pt
```

## 17. 实现索引

- [VU 数据、crop、增强、fold 和 sampler](../data.py)
- [当前全局 EfficientNet 与单调序数头](../model.py)
- [当前训练、验证和 checkpoint](../train_efficientnet.py)
- [数据索引和增强预览生成](../prepare_data.py)
- [数据测试](../tests/test_data.py)
