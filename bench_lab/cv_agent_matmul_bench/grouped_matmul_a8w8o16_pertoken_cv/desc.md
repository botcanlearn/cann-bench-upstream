# GroupedMatmul A8W8O16 per-token/per-channel 算子描述

## 1. 算子简介

GroupedMatmul 是面向大语言模型 MoE 推理的分组矩阵乘算子。输入 token 已按 expert 连续排列，算子根据 `groupList` 划分各 expert 的 token 区间，使用对应的 INT8 权重执行矩阵乘，再完成 per-channel 与 per-token 反量化并输出 BF16。

**主要应用场景**：

- MoE routed token 的 expert grouped matmul
- INT8 激活与 INT8 expert 权重的 A8W8 推理
- per-token 激活 scale 与 per-channel 权重 scale 的动态反量化

**算子特征**：

- 难度等级：L4（Grouped Contraction + Dequantization）
- 输入：INT8 `x/weight`、BF16 per-channel `scale`、FP32 `perTokenScale`
- 计算：按 expert 分组的 INT8 矩阵乘、INT32 累加、FP32 反量化
- 输出：BF16 `[M,N]`
- 数据流：CUBE 计算 INT8 x INT8 -> INT32，VECTOR 执行反量化、per-token 缩放和 BF16 Cast

本 benchmark 固定对齐 Atlas A3 `aclnnGroupedMatmulV5` 的以下路径：

```text
aclnnGroupedMatmulV5
  -> A8W8O16
  -> x per-token scale [M]
  -> weight per-channel scale [E,N]
  -> groupType=0, groupListType=0, splitItem=3
  -> GMM_QUANT_BF16
  -> GMM_CV_SPLIT_IMP(GMMQuantMixCoreCompute, GMMProcess, ...)
```

该路径使用 `ops-transformer/gmm/grouped_matmul/op_kernel` 下的 A3 C->V 实现，不使用 `arch35` 的 RegBase/MicroAPI kernel， 不覆盖 A8W8O8、A8W8O32、A4W4、A8W4、A16W8、weight NZ、转置权重、bias、offset、激活或动态输出量化路径。

## 2. 算子定义

### 数学公式

设：

- `x` 的 shape 为 `[M,K]`；
- `weight` 的 shape 为 `[E,K,N]`；
- `scale` 的 shape 为 `[E,N]`；
- `perTokenScale` 的 shape 为 `[M]`；
- `groupList` 是长度为 `E` 的 cumsum expert 边界。

令 $g_{-1}=0$，$g_i=groupList[i]$。第 $i$ 个 expert 的输入区间为：

$$
X_i=x[g_{i-1}:g_i,:]
$$

第 $i$ 个 expert 的矩阵乘结果为：

$$
Z_i=X_i \mathbin{@} weight[i,:,:]
$$

对 $Z_i$ 的每个元素，同时应用对应输出通道的 scale 和对应 token 的 scale：

$$
y[g_{i-1}+r,n]
=
\left(
\sum_{k=0}^{K-1}
x[g_{i-1}+r,k]
\times
weight[i,k,n]
\right)
\times
scale[i,n]
\times
perTokenScale[g_{i-1}+r]
$$

其中 $0 \le r < g_i-g_{i-1}$，$0 \le n < N$。

### 步骤说明

1. 按 cumsum `groupList` 将 `x` 的 M 轴划分为 E 个连续 expert 区间。
2. 对每个非空 expert 区间，将其 token 与该 expert 的权重相乘。
3. 对矩阵乘结果的每个输出通道乘对应的 per-channel scale。
4. 对结果的每一行乘对应 token 的 per-token scale。
5. 将各 expert 的结果按原区间写入输出。

相邻两个 `groupList` 值允许相等，表示对应 expert 为空。空 expert 不执行 matmul，也不改变其他 expert 的行区间。本 benchmark 固定 `groupList[-1] == M`，因此输出 `[0,M)` 的每一行均由且仅由一个非空 expert 区间写入。

## 3. 接口规范

### 算子原型

生成算子只需要实现本 benchmark 已经固定的单一路径，逻辑接口为：

```python
cann_bench.grouped_matmul(
    Tensor x,
    Tensor weight,
    Tensor scale,
    Tensor groupList,
    Tensor perTokenScale,
) -> Tensor y
```

AscendC 实现直接接收上述 5 个 Tensor，并输出一个 Tensor。本 benchmark 已固定单 Tensor 输入输出、M 轴 cumsum 分组、ND 非转置 weight、无 bias、无 offset、无激活和无 tuning 输入。无需实现通用 `aclnnGroupedMatmulV5` 的 TensorList 分发，也不得为这些固定语义增加运行时参数或分支。

### 输入参数

| 参数 | 类型 | Shape | dtype | 描述 |
|---|---|---|---|---|
| `x` | Tensor（必选） | `[M,K]` | int8 | 已按 expert 连续排列的 routed token |
| `weight` | Tensor（必选） | `[E,K,N]` | int8 | E 组 ND expert 权重，本 benchmark 固定不转置 |
| `scale` | Tensor（必选） | `[E,N]` | bfloat16 | 每个 expert、每个输出 channel 一个反量化因子 |
| `groupList` | Tensor（必选） | `[E]` | int64 | cumsum expert 分组边界 |
| `perTokenScale` | Tensor（必选） | `[M]` | float32 | 每个 token 一个激活反量化因子 |

### 输出

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `y` | `[M,N]` | BFLOAT16 | grouped matmul 反量化结果 |

### 数据类型组合

| `x` | `weight` | `scale` | `groupList` | `perTokenScale` | `y` | 量化模式 |
|---|---|---|---|---|---|---|
| INT8 | INT8 | BFLOAT16 | INT64 | FLOAT32 | BFLOAT16 | activation per-token / weight per-channel |

### 规则与约束

- `M >= 1`、`K >= 1`、`N >= 1`，且 `1 <= E <= 1024`。
- `x.shape == [M,K]`。
- `weight.shape == [E,K,N]`，且不转置。
- `scale.shape == [E,N]`，固定为 per-channel BF16 scale。
- `groupList.shape == [E]`，dtype 必须为 INT64。
- `perTokenScale.shape == [M]`，dtype 必须为 FLOAT32。
- `y.shape == [M,N]`，dtype 固定为 BFLOAT16。
- `groupList` 必须非负、单调非递减；本 benchmark 固定最后一个值等于 M。
- 相邻 cumsum 边界可以相等，表示空 expert。
- 非转置 ND 场景要求 `K < 65536`、`N < 65536`。
- `x/weight/y` 均为单 Tensor 场景，不覆盖多 TensorList 输入或输出。
- 本 benchmark 固定无 bias、无 offset、无激活、无动态输出量化。

### 支持范围

`cases.yaml` 与 `cases.csv` 一一对应，共包含 20 个 BF16 正向 case：

| 维度 / 参数 | cases 覆盖 | 备注 |
|---|---:|---|
| `M` | 1～96 | 覆盖单 token、常规批量、tail M |
| `K` | 64～1025 | 覆盖对齐 K、`K=257` 和 `K=1025` 尾块 |
| `N` | 32～1024 | 覆盖窄输出、常规输出和宽输出 |
| `E` | 1、2、3、4、5、6、7、8、16 | 覆盖单 expert 到多 expert |
| `groupList` | 均衡、非均衡、空 expert | 包含首 expert 为空、中间 expert 为空和多个空 expert |
| 输入值域 | `[-2,2]` | 控制 INT32 累加结果规模 |

最大 case 为 `M=96, K=1024, N=1024, E=4`，单 case 约包含 1.01 亿次乘加。`case11` 保留为 `M=1, K=64, N=32` 的最小 smoke case；`case3` 和 `case16` 分别使用 `K/N=257/191` 与 `1025/511` 覆盖非对齐尾块。

## 4. 精度要求

采用 [CANN 生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)

### 浮点输出 `y`

对 BF16 `y` 统计平均相对误差 MERE 和最大相对误差 MARE：

$$
\operatorname{MERE}
=
\operatorname{avg}
\left(
\frac{|actual-golden|}{|golden|+10^{-7}}
\right)
$$

$$
\operatorname{MARE}
=
\max
\left(
\frac{|actual-golden|}{|golden|+10^{-7}}
\right)
$$

`y` 的接口 dtype 固定为 BFLOAT16，精度阈值按输出 dtype 判定：

| 输出 | 判定 dtype | Threshold | 通过条件 |
|---|---|---:|---|
| `y` | BFLOAT16 | `2^-7` | `MERE < 2^-7` 且 `MARE < 10 * 2^-7` |

## 5. 标准 Golden 代码

```python
#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import torch


def get_input(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    groupList: torch.Tensor,
    perTokenScale: torch.Tensor,
    **attrs,
) -> list:
    """从 attrs.group_list_values 重建确定性的 groupList 张量。

    cases.yaml 将 cumsum 分组边界放在 group_list_values 属性里（golden 读取它），
    但被测 kernel 只看 groupList 张量。若不由 get_input 重建，groupList 会被
    value_range 随机生成（可能为负、非单调），导致 kernel 与 golden 分组不一致。
    返回值同时替换 golden 与 AI 算子的输入，确保对比公平。
    """
    gl = attrs.get("group_list_values")
    if gl is not None:
        groupList = torch.tensor(list(gl), dtype=torch.int64, device=x.device)
    return [x, weight, scale, groupList, perTokenScale]


# Atlas A3: grouped_matmul.cpp A8W8O16 -> GMM_CV_SPLIT_IMP(GMMQuantMixCoreCompute, GMMProcess).
def grouped_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    groupList: torch.Tensor,
    perTokenScale: torch.Tensor,
    group_list_values=None,
) -> torch.Tensor:
    """执行 Atlas A3 aclnnGroupedMatmulV5 的 A8W8O16 per-token grouped matmul。

    对齐的官方 kernel 路径：
        op_kernel/grouped_matmul.cpp
          -> GMM_QUANT_BF16
          -> GMM_CV_SPLIT_IMP(GMMQuantMixCoreCompute, GMMProcess, ...)

    计算语义：
        groupList 使用 cumsum 边界将 x 的 M 轴划分给 E 个 expert。
        对每个非空 expert 区间 [start, end)：
            accumulator = x[start:end].int32 @ weight[expert].int32
            dequantized = accumulator.float32 * scale[expert].float32
            output = dequantized * perTokenScale[start:end].float32
        最后将 FP32 中间结果转换为 BF16。

        该顺序对应 GMMQuantMixCoreCompute 中的 INT32 Cube 累加、
        AscendDequant、per-token FP32 Mul 和最终 Cast。本 benchmark 固定
        无 bias、无 offset、actType=0，不包含激活或动态输出量化。

    输入：
        x:
            shape 为 [M, K]、dtype 为 torch.int8 的 routed token。
            Atlas A3 路径要求 K < 65536。
        weight:
            shape 为 [E, K, N]、dtype 为 torch.int8 的 ND expert 权重；
            本 benchmark 固定不转置，且 1 <= E <= 1024、N < 65536。
        scale:
            shape 为 [E, N]、dtype 为 torch.bfloat16 的 per-expert
            per-channel 反量化因子。
        groupList:
            shape 为 [E]、dtype 为 torch.int64 的非负单调非递减
            cumsum 边界。官方算子允许最后一个值小于等于 M；本 benchmark
            固定为 M，以保证输出的每一行都有定义。
            相邻值可以相等，表示对应 expert 为空。
        perTokenScale:
            shape 为 [M]、dtype 为 torch.float32 的 per-token 因子。

    Benchmark 辅助参数：
        group_list_values:
            runner 用于构造确定性 groupList 的辅助值，不是 ACLNN 参数。
            为 None 时直接读取 groupList Tensor。

    固定场景：
        本函数直接表达单 Tensor 输出、M 轴分组、cumsum groupList 和
        无激活语义，因此不再暴露只负责选路的 ACLNN 属性。

    输出：
        shape 为 [M, N]、dtype 为 torch.bfloat16。最终转换与官方
        ST reference 一致，使用 PyTorch .to(torch.bfloat16) 表达。

    典型 case：
        - 常规：x=[16,128]，weight=[2,128,64]，groupList=[8,16]。
        - K/N tail：x=[18,1025]，weight=[4,1025,511]，
          groupList=[4,9,13,18]。

    完整测试集合见同目录 cases.yaml 和 cases.csv。
    """
    m = x.shape[0]
    n = weight.shape[2]
    groups = group_list_values
    if groups is None:
        groups = groupList.detach().to(device="cpu").tolist()

    out = torch.zeros(m, n, dtype=torch.float32, device=x.device)

    start = 0
    for expert_id, end in enumerate(groups):
        if end > start:
            accumulator = torch.matmul(
                x[start:end].to(torch.int32),
                weight[expert_id].to(torch.int32),
            )
            dequantized = accumulator.to(torch.float32) * scale[expert_id]
            out[start:end] = dequantized * perTokenScale[start:end, None]
        start = end

    return out.to(torch.bfloat16)
```

`group_list_values` 仅用于让 benchmark runner 为 `groupList` 注入确定的 cumsum 数值，不是 AscendC kernel 输入；该参数为 `None` 时，Golden 直接读取 `groupList` Tensor。

Golden 保留算子计算语义所需的 dtype 转换：INT8 输入转 INT32 表达硬件累加，累加结果转 FP32 后依次乘 per-channel 和 per-token scale，最后转换为 BF16。

上述顺序不可改写为先合并两个 scale 再乘 INT32 累加结果；虽然实数公式等价，但 FP32 乘法顺序变化可能造成不同的舍入结果。

## 6. 额外信息

### Golden 调用示例

```python
import torch

from golden import grouped_matmul

M, K, N, E = 16, 128, 64, 2
groups = [8, 16]

x = torch.randint(-2, 3, (M, K), dtype=torch.int8)
weight = torch.randint(-2, 3, (E, K, N), dtype=torch.int8)
scale = torch.rand((E, N), dtype=torch.bfloat16)
group_list = torch.tensor(groups, dtype=torch.int64)
per_token_scale = torch.rand((M,), dtype=torch.float32)

y = grouped_matmul(
    x,
    weight,
    scale,
    group_list,
    per_token_scale,
    group_list_values=groups,
)

assert y.shape == (M, N)
assert y.dtype == torch.bfloat16
```

### Case 设计

`cases.csv` 的 20 个 case 分为以下几类：

- 基础与边界：单 token、单 expert、少 expert 和多 expert。
- 分组分布：均衡、非均衡、首 expert 为空、中间 expert 为空、多个 expert 为空。
- 对齐覆盖：常见 128/256/512 对齐 K/N，以及 257/191、1025/511 非对齐尾块。
- 规模覆盖：保留小型 smoke case，同时将最大规模扩展到 `M96 x K1024 x N1024`。
- 维度扩展：保持 M、E 和 groupList 语义不变，仅扩大 K/N。

### 实现对齐依据

- **aclnnGroupedMatmulV5.md**：A3/A2 A8W8 dtype 组合、动态 K-C scale、groupType 和 groupList 约束
- **aclnn_grouped_matmul_v5.h**：`aclnnGroupedMatmulV5` ACLNN 参数原型
- **grouped_matmul.cpp**：A8W8O16 的 `GMM_QUANT_BF16` 与 `GMM_CV_SPLIT_IMP` 分发
- **grouped_matmul_quant_mixcore.h**：`AscendDequant`、per-token FP32 `Mul` 和最终 BF16 `Cast`
- **grouped_matmul_tiling.cpp**：static tiling、AIV/AIC 比例、fixed-axis 和 pretiling 的 Host 选路条件
- **executor_aclnnGroupedMatmulV5_A8W8O16.py**：官方 PyTorch reference 的 INT32 matmul 与 FP32 反量化顺序
