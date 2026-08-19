# BevPool 算子 API 描述

## 1. 算子简介

BEVPoolV2 风格的视锥特征池化算子，是 LSS（Lift-Splat-Shoot）范式 BEV 感知网络中 view transformation 的核心步骤。多相机图像特征经 DepthNet 预测出逐像素深度分布后，图像特征与深度权重的外积构成视锥点云；BevPool 将每个视锥点的加权特征聚合（累加）到其落入的 BEV（鸟瞰图）网格中，完成从透视视角到 BEV 视角的转换。

BEVPoolV2 的关键优化是**预计算索引**：视锥到 BEV 的几何映射只依赖相机内外参，可离线算好三组线性索引（ranks_depth、ranks_feat、ranks_bev），在线阶段算子退化为纯粹的 gather-乘-scatter-add，无需物化视锥点云本身。

**主要应用场景**：
- BEVDet / BEVDet4D / BEVFusion 等多相机 BEV 3D 目标检测的 view transformation
- 自动驾驶多传感器融合中相机分支的 BEV 特征构建
- 占用网络（occupancy network）等需要视锥到体素聚合的感知任务

**算子特征**：
- 难度等级：L4（FusedComposite）
- 五输入（depth、feat、三组 int32 索引）单输出，融合两路 gather、逐点乘与 scatter-add
- **ranks_bev 无序且大量重复**：多个视锥点落入同一 BEV 网格，并行实现存在写冲突，kernel 需用原子累加或"按 ranks_bev 排序 + 分段归约"策略——这是本算子的主要难度来源
- 纯访存密集型（每点仅一次乘法），索引间接寻址的访存局部性决定性能
- 累加全程 fp32，输出转回输入 dtype

## 2. 算子定义

### 数学公式

将 depth 展平为一维、feat 展平为二维行主序矩阵：

$$
\text{depth}^{flat} \in \mathbb{R}^{N_{cam} \cdot D \cdot fH \cdot fW}, \qquad
\text{feat}^{flat} \in \mathbb{R}^{(N_{cam} \cdot fH \cdot fW) \times C}
$$

对每个视锥点 $i \in [0, P)$：

$$
\text{out}^{flat}[\text{ranks\_bev}[i],\, :] \mathrel{+}= \text{depth}^{flat}[\text{ranks\_depth}[i]] \cdot \text{feat}^{flat}[\text{ranks\_feat}[i],\, :]
$$

未被任何点命中的 BEV 网格保持 0。

### 计算子步骤

1. **gather 深度权重**：$w_i = \text{depth}^{flat}[\text{ranks\_depth}[i]]$，标量（每点一个深度概率）
2. **gather 特征行**：$v_i = \text{feat}^{flat}[\text{ranks\_feat}[i], :] \in \mathbb{R}^C$（同一特征行被 D 个深度格点重复引用）
3. **逐点乘权**：$c_i = w_i \cdot v_i$
4. **scatter-add 聚合**：$\text{out}^{flat}[\text{ranks\_bev}[i]] \mathrel{+}= c_i$，fp32 累加
5. **输出**：reshape 为 [bev_h, bev_w, C] 并转回输入 dtype

### 写冲突与并行化特点

- ranks_bev 无序且可重复（真实场景中平均每个 BEV 网格被十几到上百个视锥点命中），朴素按点并行会产生**同地址写冲突**
- 可行策略：(a) 原子累加（AtomicAdd）；(b) 预排序 ranks_bev + 分段归约（BEVPoolV2 官方 CUDA 实现的 interval 方案）；(c) 按 BEV 网格分桶、桶内串行
- fp32 累加下不同累加顺序的舍入差异极小，精度校验按标准阈值执行
- feat 的 gather 具有 D 重复用（同一像素的 C 维特征行被该像素全部深度格点引用），片上缓存可显著减少重复搬运

## 3. 接口规范

### 算子原型

```python
bev_pool(Tensor depth, Tensor feat, Tensor ranks_depth, Tensor ranks_feat, Tensor ranks_bev, int bev_h, int bev_w) -> Tensor bev_feat
```

### 输入参数说明

| 参数 | 类型 | 必需 | dtype | shape | 描述 |
|------|------|------|-------|-------|------|
| depth | Tensor | 是 | float16/float32 | [N_cam, D, fH, fW] | 视锥深度分布预测（DepthNet softmax 输出，取值 [0, 1]） |
| feat | Tensor | 是 | 与 depth 一致 | [N_cam, fH, fW, C] | 相机图像特征（channel-last） |
| ranks_depth | Tensor | 是 | int32 | [P] | depth 展平后的线性索引，取值 [0, N_cam·D·fH·fW − 1] |
| ranks_feat | Tensor | 是 | int32 | [P] | feat 展平为 [N_cam·fH·fW, C] 后的行索引，取值 [0, N_cam·fH·fW − 1] |
| ranks_bev | Tensor | 是 | int32 | [P] | BEV 网格线性索引，取值 [0, bev_h·bev_w − 1]，无序可重复 |
| bev_h | int | 是 | - | 标量 | BEV 网格高度 |
| bev_w | int | 是 | - | 标量 | BEV 网格宽度 |

### 输出

| 参数 | dtype | shape | 描述 |
|------|-------|-------|------|
| bev_feat | 与 depth/feat 一致 | [bev_h, bev_w, C] | BEV 网格特征，未命中网格为 0 |

### 数据类型

| depth/feat dtype | ranks dtype | 输出 dtype | 内部累加 |
|------------------|------------|-----------|---------|
| float16 | int32 | float16 | fp32 |
| float32 | int32 | float32 | fp32 |

### 规则与约束

- depth 与 feat 的 dtype 必须一致，且第一维（N_cam）相同；feat 的 fH、fW 与 depth 的后两维一致
- ranks_depth、ranks_feat、ranks_bev 长度必须相同（均为 P），dtype 固定 int32
- 三组索引的取值必须落在各自合法值域内（评测框架按 value_range 独立随机生成，越界由 value_range 排除）：
  - ranks_depth ∈ [0, N_cam·D·fH·fW − 1]
  - ranks_feat ∈ [0, N_cam·fH·fW − 1]
  - ranks_bev ∈ [0, bev_h·bev_w − 1]
- 三组索引之间无排序、无对应关系约束：ranks_bev 无序且可重复，kernel 必须正确处理同一 BEV 网格的并发累加（原子加或排序聚合）
- 累加必须在 fp32 精度下进行，最终转回输入 dtype；未被命中的 BEV 网格输出 0
- 真实部署中 ranks 由相机几何离线预计算（含视锥点过滤，P ≤ N_cam·D·fH·fW）；本算子将其视为任意合法索引输入

### 支持范围

| 维度 / 参数 | 支持值 | 备注 |
|---|---|---|
| `N_cam`（相机数） | 6 | nuScenes 环视配置 |
| `D`（深度格点数） | {59, 118} | BEVDet-R50（1~60m 步长 1m）/ BEVFusion 高分辨率配置 |
| `fH × fW`（特征图） | 16×44 / 32×88 | 输入分辨率 256×704 / 512×1408 下采样 16 倍 |
| `C`（特征通道） | {64, 80, 128} | cases.csv 三种均覆盖 |
| `P`（视锥点数） | 50000 ~ 1993728 | 与 N_cam·D·fH·fW 同量级（几何过滤后约 40%~100%） |
| `bev_h × bev_w` | 128×128 / 256×256 | 感知范围 51.2m、网格 0.8m / 0.4m（或 102.4m、0.4m） |
| `depth` 取值 | [0, 1] | softmax 深度概率 |
| `feat` 取值 | [-1, 1] 典型 | |
| dtype | float16 / float32 | cases.csv 两种均覆盖 |

## 4. 精度要求

采用[生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)进行验证。

**误差指标**：

1. 平均相对误差（MERE）：采样点中相对误差平均值

   $$
   \text{MERE} = \text{avg}(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

2. 最大相对误差（MARE）：采样点中相对误差最大值

   $$
   \text{MARE} = \max(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

**通过标准**：

| 数据类型 | FLOAT16 | BFLOAT16 | FLOAT32 | HiFLOAT32 | FLOAT8 E4M3 | FLOAT8 E5M2 |
|----------|---------|----------|---------|-----------|-------------|-------------|
| **通过阈值(Threshold)** | 2^-10 | 2^-7 | 2^-13 | 2^-11 | 2^-3 | 2^-2 |

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过。

## 5. 标准 Golden 代码

```python
import torch


def _bev_pool_core(depth, feat, ranks_depth, ranks_feat, ranks_bev, bev_h, bev_w, compute_dtype):
    """核心计算：以 compute_dtype 精度执行 gather-乘-scatter-add。"""
    C = feat.shape[-1]
    depth_flat = depth.reshape(-1).to(compute_dtype)               # [N_cam*D*fH*fW]
    feat_flat = feat.reshape(-1, C).to(compute_dtype)              # [N_cam*fH*fW, C]

    # gather: 深度权重 [P] 与特征行 [P, C]
    w = depth_flat[ranks_depth.long()]                             # [P]
    v = feat_flat[ranks_feat.long()]                               # [P, C]
    contrib = w.unsqueeze(-1) * v                                  # [P, C]

    # scatter-add: 按 ranks_bev 累加到 BEV 网格
    bev = torch.zeros(bev_h * bev_w, C, dtype=compute_dtype, device=feat.device)
    bev.index_add_(0, ranks_bev.long(), contrib)
    return bev.reshape(bev_h, bev_w, C)


def bev_pool(
    depth: torch.Tensor,
    feat: torch.Tensor,
    ranks_depth: torch.Tensor,
    ranks_feat: torch.Tensor,
    ranks_bev: torch.Tensor,
    bev_h: int,
    bev_w: int,
) -> torch.Tensor:
    """
    BEVPoolV2 风格视锥特征池化（plain golden = bench：fp32 累加）

    Args:
        depth: [N_cam, D, fH, fW] 视锥深度分布预测, float16/float32
        feat: [N_cam, fH, fW, C] 相机图像特征 (channel-last), dtype 与 depth 一致
        ranks_depth: [P] depth 展平后的线性索引 (int32), 取值 [0, N_cam*D*fH*fW - 1]
        ranks_feat: [P] feat 展平为 [N_cam*fH*fW, C] 后的行索引 (int32), 取值 [0, N_cam*fH*fW - 1]
        ranks_bev: [P] BEV 网格线性索引 (int32), 取值 [0, bev_h*bev_w - 1], 无序可重复
        bev_h: BEV 网格高度
        bev_w: BEV 网格宽度

    Returns:
        bev_feat: [bev_h, bev_w, C] BEV 网格特征, dtype 与输入一致
    """
    bev = _bev_pool_core(depth, feat, ranks_depth, ranks_feat, ranks_bev, bev_h, bev_w, torch.float32)
    return bev.to(feat.dtype)


def bev_pool_oracle(
    depth: torch.Tensor,
    feat: torch.Tensor,
    ranks_depth: torch.Tensor,
    ranks_feat: torch.Tensor,
    ranks_bev: torch.Tensor,
    bev_h: int,
    bev_w: int,
) -> torch.Tensor:
    """Oracle (g)：dtype-agnostic，计算精度跟随输入（fp64_cpu 下为 fp64 真值），不硬编码 .float()。"""
    return _bev_pool_core(depth, feat, ranks_depth, ranks_feat, ranks_bev, bev_h, bev_w, feat.dtype)
```

## 6. 额外信息

### 算子调用示例

```python
import torch

N_cam, D, fH, fW, C = 6, 59, 16, 44, 64   # BEVDet-R50 配置
bev_h, bev_w = 128, 128
P = 200000

depth = torch.rand(N_cam, D, fH, fW, dtype=torch.float16, device="npu")
feat = torch.randn(N_cam, fH, fW, C, dtype=torch.float16, device="npu")
ranks_depth = torch.randint(0, N_cam * D * fH * fW, (P,), dtype=torch.int32, device="npu")
ranks_feat = torch.randint(0, N_cam * fH * fW, (P,), dtype=torch.int32, device="npu")
ranks_bev = torch.randint(0, bev_h * bev_w, (P,), dtype=torch.int32, device="npu")

bev_feat = bev_pool(depth, feat, ranks_depth, ranks_feat, ranks_bev, bev_h, bev_w)
# bev_feat.shape: [bev_h, bev_w, C]
```
