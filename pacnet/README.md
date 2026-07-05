# PACNet: Plant-Adaptive Counting Network

> **模式识别与机器学习课程设计** —— 针对植物特点的类别无关计数算法改进
>
> 基于三个基线模型（CLIP-Count、CountingDINO、CountGD）的评估痛点分析，
> 结合 TasselNet [3] 的局部计数回归思想和 TasselNetV4 [2] 的多尺度基础模型架构，
> 提出的针对植物场景的适应性计数网络。

---

## 目录

- [痛点分析与改进动机](#痛点分析与改进动机)
- [算法设计思路](#算法设计思路)
- [网络架构](#网络架构)
- [创新点详解](#创新点详解)
- [实验结果](#实验结果)
- [文件结构](#文件结构)
- [快速开始](#快速开始)
- [参考文献](#参考文献)

---

## 痛点分析与改进动机

### 三个基线在植物图像上的核心痛点

通过对 CLIP-Count、CountingDINO、CountGD 在 TPC-268 数据集上的全面评估，
我们识别出以下关键问题：

#### 1. 尺度变化极端（3-4个数量级）

| 尺度 | 示例 | 尺寸范围 | 对应 FPN 层级 |
|------|------|---------|-------------|
| 微观 | 气孔、种子 | ~0.1-1mm | P2 (1/4) |
| 中观 | 花朵、叶片 | ~1-10cm | P3 (1/8) |
| 宏观 | 果实、植株 | ~10-100cm | P4-P5 (1/16-1/32) |

- **CLIP-Count**: 单一尺度的密度图无法同时捕捉微小气孔和大果实
- **CountingDINO**: 固定 patch 尺寸的 DINO 特征匹配在不同尺度下失效
- **CountGD**: GroundingDINO 的检测框尺度范围有限

#### 2. 领域偏移（Domain Shift）

所有三个基线均在 FSC-147（通用物体）上预训练/设计，植物图像具有：
- **非刚性形态**: 植物器官缺乏刚性边界，形状变化极大
- **低对比度**: 绿色植物在绿色背景中，前景-背景对比度极低
- **高度重叠**: 密集生长的植物器官间特征重叠严重

#### 3. 密度图系统性偏差

| 模型 | 偏差方向 | 偏差幅度 | 根因 |
|------|---------|---------|------|
| CLIP-Count | 低估 | 64.59 vs 69.31 (-6.8%) | 密度图在密集区域被压缩 |
| CountingDINO | 高估 | 70.20 vs 65.76 (+6.7%) | DINOv2 特征对纹理敏感产生虚假匹配 |
| CountGD | 强低估 | 49.89 vs 65.79 (-24.2%) | GroundingDINO 植物漏检率高 |

### 改进策略

参考 [3] TasselNet（局部计数回归）和 [2] TasselNetV4（多尺度基础模型），
我们提出以下改进：

```
改进策略矩阵：
┌─────────────────┬────────────┬────────────┬────────────┐
│ 痛点            │ TasselNet  │ TasselNetV4│ PACNet     │
│                 │ 解法 [3]   │ 解法 [2]   │ 本工作     │
├─────────────────┼────────────┼────────────┼────────────┤
│ 尺度变化极端     │ 滑动窗口   │ FPN + ViT  │ FPN + 自适应 │
│                 │ 多尺度     │ 多尺度交叉  │ 注意力头   │
├─────────────────┼────────────┼────────────┼────────────┤
│ 领域偏移         │ ImageNet   │ 大规模     │ 植物增强   │
│                 │ 预训练     │ 预训练     │ + 全局上下文 │
├─────────────────┼────────────┼────────────┼────────────┤
│ 密度图偏差       │ 局部计数   │ 密度回归   │ 全局+局部  │
│                 │ 回归       │ +检测      │ 混合架构   │
└─────────────────┴────────────┴────────────┴────────────┘
```

---

## 算法设计思路

### 核心思想：多尺度自适应 + 局部精细化

PACNet 的设计基于两个关键观察：

1. **TasselNet [3] 的启示**: 植物计数不需要理解"这是什么"，只需要知道"这里有多少"。
   局部计数回归通过滑动窗口聚合局部密度，天然适合形态多变的植物场景。

2. **TasselNetV4 [2] 的启示**: 跨尺度、跨场景、跨物种的通用计数能力需要多尺度特征
   表示和自适应尺度选择机制。

### 算法流程

```
输入图像 (H x W x 3)
    │
    ▼
┌─────────────────────────────────────────────┐
│ 步骤 1: 多尺度特征提取 (ResNet-50 + FPN)      │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐       │
│  │  P2  │ │  P3  │ │  P4  │ │  P5  │       │
│  │ 1/4  │ │ 1/8  │ │ 1/16 │ │ 1/32 │       │
│  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘       │
│     │        │        │        │            │
│     └────────┴───┬────┴────────┘            │
│                  ▼                           │
│     ┌──────────────────────┐                 │
│     │  Global Context      │                 │
│     │  Modulation          │                 │
│     └──────────┬───────────┘                 │
└────────────────┼─────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────┐
│ 步骤 2: 尺度自适应密度头                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐    │
│  │ 小尺度头  │ │ 中尺度头  │ │ 大尺度头  │    │
│  │ (P2 1/4) │ │ (P3 1/8) │ │ (P4 1/16)│    │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘    │
│       │            │            │           │
│       └──────┬─────┴─────┬──────┘           │
│              ▼           ▼                   │
│     ┌──────────────────────────┐             │
│     │  Scale Attention Fusion  │             │
│     │  (自适应加权融合)         │             │
│     └────────────┬─────────────┘             │
└──────────────────┼───────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│ 步骤 3: 局部计数精细化 (TasselNet 启发)       │
│  ┌─────────────────────────────────────┐    │
│  │ 密度置信度评估 → 高密度区提取 → 局部修正  │    │
│  └─────────────────────────────────────┘    │
│                   ▼                          │
│  ┌─────────────────────────────────────┐    │
│  │ 修正项 = 局部计数 × (1 - 置信度)      │    │
│  │ 最终密度 = 全局密度 + 修正项           │    │
│  └─────────────────────────────────────┘    │
└──────────────────┼───────────────────────────┘
                   ▼
最终密度图 -> 积分 -> 计数
```

---

## 网络架构

### 1. Backbone: ResNet-50 + Feature Pyramid Network (FPN)

- **Bottom-up**: ResNet-50 提取 C2-C5 特征
- **Top-down**: FPN 横向连接生成 P2-P5 多尺度特征
- **全局上下文**: 自适应平均池化 + 全连接层生成全局调制信号
- **参数量**: ~25.6M (backbone) + ~3.5M (FPN)

### 2. Scale-Adaptive Density Heads

| 头 | 输入尺度 | 感受野 | 适用目标 |
|----|--------|--------|---------|
| 小尺度头 (Small) | P2 (1/4) | 细粒度 | 气孔、种子、小花朵 |
| 中尺度头 (Medium) | P3 (1/8) | 中等 | 叶片、中等花朵 |
| 大尺度头 (Large) | P4 (1/16) | 全局 | 果实、整株植物 |

- **每个头**: 4层 Conv-BN-ReLU → 1x1 Conv → 密度图
- **自适应融合**: 全局平均池化 → 注意力权重 → 加权求和

### 3. Local Refinement Module (TasselNet 启发)

- **动机**: 解决密度图在密集区域的系统性低估 (CLIP-Count -6.8%, CountGD -24.2%)
- **方法**: 从特征图直接回归局部密度修正，置信度自适应加权
- **灵感来源**: TasselNet [3] 的局部计数回归思想

### 4. Hybrid Training Loss

`
L = L_density + 0.1 * L_count + 0.05 * L_local
`

| 损失项 | 类型 | 作用 |
|--------|------|------|
| L_density | 加权MSE (归一化+直接+加权) | 密度图分布和幅值回归 |
| L_count | L1 (MAE) | 全局计数一致性 |
| L_local | 加权局部MSE | 高密度区精细修正 |
| L_aux | MSE | 多尺度辅助监督 |

---

## 创新点详解

### 创新点 1: 多尺度自适应密度头

**问题**: CLIP-Count 使用单尺度密度图，无法同时处理从气孔到果实的跨4个数量级的尺度变化。

**解法**: 设计3个并行密度头，每个头专注于特定尺度范围，通过可学习的注意力权重自适应融合。

对比分析：
- CLIP-Count: 单头 → 单尺度密度图
- CountingDINO: 固定patch尺寸 → 多尺度patch (divide_et_impera)
- PACNet: 多头自适应 → 可学习尺度选择
- TasselNetV4 [2]: FPN + ViT → 交叉注意力融合

### 创新点 2: 局部计数精细化

**问题**: 密度图方法在密集区域系统性低估（CLIP-Count 低估6.8%, CountGD 低估24.2%）。

**解法**: 受 TasselNet [3] 启发，增加局部计数修正分支，在高密度/低置信度区域施加修正。

对比分析：
- TasselNet [3]: 滑动窗口 + CNN → 纯局部回归
- CLIP-Count: 纯全局密度图 → 无局部修正
- CountGD: 检测 + 分割 → 漏检无法修正
- PACNet: 全局密度 + 局部修正 → 混合架构

### 创新点 3: 植物感知增强

**问题**: 所有基线在 FSC-147（通用物体）上训练/设计，植物图像域偏移严重。

**解法**: 
- 全局上下文调制 (Global Context Modulation)
- 尺度抖动增强 (Scale Jitter: 0.5x-2.0x)
- 颜色抖动和随机遮挡

---

## 实验结果

### 与三个基线的性能对比 (TPC-268 测试集)

| 指标 | CLIP-Count | CountingDINO | CountGD | PACNet (ours) |
|------|:----------:|:------------:|:-------:|:-------------:|
| **MAE** ↓ | 26.84 | 36.81 | 26.10 | 43.19 |
| **RMSE** ↓ | 52.54 | 86.90 | 59.19 | 75.63 |
| **R2** ↑ | **0.59** | -0.30 | 0.40 | -0.14 |
| **rMAE** ↓ | 0.39 | 0.56 | 0.40 | 0.73 |
| **rRMSE** ↓ | 0.76 | 1.32 | 0.90 | 1.28 |
| **参数量** | ~350M | 87M | ~1B+ | ~30M |
| **FLOPS** | ~300G | ~2130G | ~3500G | ~20G |
| **计数范式** | 文本引导密度图 | 样例引导匹配 | 检测+分割 | 多尺度密度+局部修正 |

> **注**: PACNet 使用 500 张训练集、15 个 epoch、batch_size=4 在 CPU 上训练。
> 当前结果显示 MAE=43.19，R2=-0.14，模型仍未充分收敛。
> 建议增加 epoch 数或使用 GPU 训练以获取更好效果。

### 预期改进

基于设计分析和痛点对应，预期改进幅度：

| 痛点 | PACNet 对策 | 预期改善 |
|------|-----------|---------|
| 尺度变化极端 | 多尺度 FPN + 自适应头 | MAE ↓15-25% |
| 密度图低估 | 局部精细化修正 | 偏差从-6.8%降至-3% |
| 领域偏移 | 植物增强 + context | 对低对比度图像更鲁棒 |
| 计算效率 | 轻量 backbone (~30M) | FLOPS 比 CountGD ↓90% |

---

## 文件结构

```
pacnet/
├── models/
│   ├── __init__.py
│   ├── pacnet.py           # 主模型 (PACNet)
│   ├── backbone.py         # ResNet-50 + FPN 骨干网络
│   ├── multi_scale_head.py # 尺度自适应密度头
│   ├── local_refine.py     # 局部精细化模块 (TasselNet启发)
│   └── loss.py             # 混合损失函数
├── datasets/
│   ├── __init__.py
│   └── plant_dataset.py    # TPC-268 数据集加载 + 植物增强
├── utils/
│   ├── __init__.py
│   ├── metrics.py          # 评估指标 (MAE/RMSE/R2/rMAE/rRMSE)
│   ├── config.py           # 配置文件
│   └── visualization.py    # 可视化工具
├── config/
│   └── default.yaml        # 默认配置入口
├── train.py                # 训练脚本
├── evaluate.py             # 评估脚本
├── inference.py            # 单图推理
├── requirements.txt        # 依赖
└── README.md               # 本文件
```

---

## 快速开始

### 环境要求

`ash
# 使用已有环境 (见 plant_counting_env/venv)
# 或创建新环境
pip install -r pacnet/requirements.txt
`

### 评估

`ash
python pacnet/evaluate.py \
    --num_images 100 \
    --output_dir results/pacnet

# 使用多尺度测试增强 (精度更高但速度更慢)
python pacnet/evaluate.py \
    --num_images 100 \
    --multi_scale \
    --output_dir results/pacnet_ms
`

### 训练

`ash
python pacnet/train.py \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-4 \
    --output_dir results/pacnet
`

### 单图推理

`ash
python pacnet/inference.py path/to/image.jpg \
    --weights results/pacnet/best_model.pth \
    --output output_vis.png
`

---

## 参考文献

[1] Yiming Lin et al. "CLIP-Count: Towards Text-Guided Object Counting." CVPR 2024.

[2] Xiaonan Hu et al. "TasselNetV4: A vision foundation model for cross-scene, cross-scale,
    and cross-species plant counting." ISPRS Journal of Photogrammetry and Remote Sensing,
    231, 745-760, 2026.

[3] Hao Lu et al. "TasselNet: counting maize tassels in the wild via local counts regression
    network." Plant Methods, 13(1), 79, 2017.

[4] Niki Amini-Naieni et al. "CountGD: Multi-Modal Open-World Counting." NeurIPS 2024.

[5] Lore Bianchi et al. "CountingDINO: A Training-free Pipeline for Class-Agnostic Counting
    using Unsupervised Backbones." WACV 2026.

[6] Jinyu Xu et al. "Plant Taxonomy Meets Plant Counting: A Fine-Grained, Taxonomic Dataset
    for Counting Hundreds of Plant Species." CVPR 2026.

