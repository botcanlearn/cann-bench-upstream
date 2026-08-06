# GroupedMatmulFinalizeRouting WeightNzV2 A8W8 per-token/per-channel 算子描述

## 1. 算子简介

GroupedMatmulFinalizeRouting 是面向大语言模型 MoE 推理的融合算子，将按专家排列的量化 token 分组执行矩阵乘和反量化，再依据路由权重与目标行号完成专家结果聚合，同时合入共享专家输出。

**主要应用场景**：

- MoE 专家 INT8 权重与 INT8 激活的 grouped matmul
- per-token 激活反量化与 per-channel 权重反量化
- Top-K 专家结果的 finalize routing 与 scatter-add
- 共享专家输出与路由专家输出的融合

**算子特征**：

- 难度等级：L4（Grouped Contraction + Dequantization + Indexed Reduction）
- 输入：INT8 激活、INT8 专家权重、FP32 scale、BF16 sharedInput
- 计算：分组矩阵乘、FP32 反量化、logit 加权和 rowIndex 聚合
- 输出：FP32 `[output_bs,7168]`
- **强制约束——权重布局：真实算子接口的 `x2` 必须使用 FRACTAL_NZ；Golden 和 cases 中的 `[E,2048,7168]` 仅表示等价的逻辑矩阵**

本 benchmark 固定 Atlas A2/A3 的下列路径：

```text
aclnnGroupedMatmulFinalizeRoutingWeightNzV2
  -> A8W8
  -> per-token scale [M]
  -> per-channel scale [E,7168]
  -> rowIndex INT64
  -> groupListType=0
  -> GMMFR_A8W8_IMPL(int64_t, float, true, false)
```

不覆盖 A8W4、MX 量化、BF16 scale、INT32 rowIndex、无 sharedInput、`groupListType=1` 或其他平台路径。

> **重要提醒**
>
> - **唯一 AscendC 实现目标**：`grouped_matmul_finalize_routing`，实现以该函数的输入、输出和数学语义为准。
> - **Benchmark 输入准备**：runner 在调用 Golden 前根据 case 元数据写入 `groupList` 和 `rowIndex` Tensor，并设置确定性运行模式；这些元数据不是 Golden 或 AscendC 算子参数。
> - **强制约束**：调用 AscendC 算子前，必须将 `x2` 转换为 FRACTAL_NZ。

## 2. 算子定义

### 数学公式

`x1` 的 M 行已经按专家连续排列。`groupListType=0` 时，`groupList[i]` 是第 i 个专家对应区间的累积结束位置：

$$
[start_i,end_i)
=
\begin{cases}
[0,groupList[0]), & i=0 \\
[groupList[i-1],groupList[i]), & i>0
\end{cases}
$$

第 i 个专家执行 INT8 矩阵乘并按 INT32 累加：

$$
acc_i
=
\operatorname{MatMul}_{INT32}
\left(
x1[start_i:end_i,:],
x2[i,:,:]
\right)
$$

使用该专家的 per-channel scale、每条路由记录的 per-token scale 和上游 logit 得到路由结果：

$$
routed[t,n]
=
\operatorname{FP32}(acc[t,n])
\cdot scale[expert(t),n]
\cdot pertokenScaleOptional[t]
\cdot logit[t]
$$

共享专家分支只写入 `[sharedInputOffset, sharedInputOffset+L)`：

$$
shared[b,n]
=
\begin{cases}
\operatorname{FP32}(sharedInput[b-sharedInputOffset,n])
\cdot sharedInputWeight, & sharedInputOffset \le b < sharedInputOffset+L \\
0, & \text{其他}
\end{cases}
$$

最终输出按 `rowIndex` 执行 scatter-add：

$$
out[b,n]
=
shared[b,n]
+
\sum_{t:rowIndex[t]=b} routed[t,n]
$$

### 步骤说明

1. 根据 cumsum `groupList` 将 `x1` 划分为 E 个专家区间，重复边界表示空专家。
2. 每个专家计算 `INT8 x INT8 -> INT32` grouped matmul。
3. 使用 `[E,7168]` 的 `scale` 完成 per-channel 反量化。
4. 使用 `[M]` 的 `pertokenScaleOptional` 和 `logit` 完成逐路由记录缩放；算子不对 `logit` 执行 softmax 或归一化。
5. 将 BF16 `sharedInput` 转入 FP32 聚合路径并乘 `sharedInputWeight`。
6. 逐条读取 `rowIndex[t]`，将 `routed[t,:]` 累加到对应输出行。

`sharedInputOffset` 只决定 sharedInput 的放置区间，**不得叠加到 rowIndex**。
`M` 无须能被 `output_bs` 整除，各输出行在 `rowIndex` 中的出现次数也可以不同。

## 3. 接口规范

### 算子 Golden 原型

```text
grouped_matmul_finalize_routing(
    Tensor x1,
    Tensor x2,
    Tensor scale,
    Tensor pertokenScaleOptional,
    Tensor groupList,
    Tensor sharedInput,
    Tensor logit,
    Tensor rowIndex,
    int output_bs,
    float sharedInputWeight=1.0,
    int sharedInputOffset=0,
) -> Tensor out
```

`grouped_matmul_finalize_routing` 同时是唯一 Golden 入口和 AscendC 实现目标。`groupListType=0` 是本路径的固定条件，不作为 Golden 或 case 的可变参数。

### Tensor 输入

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `x1` | `[M,2048]` | INT8 | 已按专家连续排列的路由激活 |
| `x2` | `[E,2048,7168]` | INT8 | Golden/cases 使用的逻辑矩阵；真实算子调用时必须转换为 FRACTAL_NZ |
| `scale` | `[E,7168]` | FLOAT32 | 每个专家、每个输出通道一个反量化因子 |
| `pertokenScaleOptional` | `[M]` | FLOAT32 | 每条路由记录一个激活反量化因子 |
| `groupList` | `[E]` | INT64 | cumsum 专家分组边界 |
| `sharedInput` | `[L,7168]` | BFLOAT16 | 共享专家输出 |
| `logit` | `[M]` | FLOAT32 | 上游计算好的路由权重 |
| `rowIndex` | `[M]` | INT64 | 每条路由记录在 out 中的 scatter-add 目标行 |

### `x2` FRACTAL_NZ 强制约束

`cases.yaml`、`cases.csv` 和 Golden 使用逻辑 shape 为 `[E,K,N]` 的 ND `x2`，目的是直接表达矩阵乘语义。该逻辑表示不改变 WeightNzV2 的真实接口契约：`x1` 保持 ND，`x2` 必须以 FRACTAL_NZ 格式传入算子。

**生成的 AscendC 算子及其 PyTorch 调用适配层必须满足以下约束：**

1. 先将逻辑 INT8 `x2` 放到 NPU。
2. 调用 `torch_npu.npu_format_cast(x2_npu, 29)` 转换为 FRACTAL_NZ，其中格式编号 `29` 对应 `ACL_FORMAT_FRACTAL_NZ`。
3. 将转换后的 `x2_nz` 传给 AscendC 算子；Kernel 读取的权重必须是 FRACTAL_NZ 存储。
4. 禁止把 ND `x2` 直接传给本路径，也禁止在 Kernel 中将 ND 内存按 NZ 布局解释。
5. `reshape`、`view`、`transpose` 或 `contiguous` 不能替代格式转换。

PyTorch 调用适配层应包含等价于以下代码的强制转换：

```python
import torch_npu

x1_npu = x1.to("npu")  # x1 保持 ND
x2_npu = x2.to("npu")
x2_nz = torch_npu.npu_format_cast(x2_npu, 29)
out = custom_op(x1_npu, x2_nz, ...)
```

`torch_npu.npu_format_cast` 属于 Host/Python 输入准备层；AscendC Kernel 接收到的 `x2` 已经是 FRACTAL_NZ。Golden 不执行该物理布局转换，因为布局变化不改变逻辑数值。性能计时只覆盖 `grouped_matmul_finalize_routing`，不得把权重格式转换计入 Kernel 耗时。

### 标量参数与输出形状

| 参数 | 类型 | 描述 |
|---|---|---|
| `output_bs` | int | Golden 创建 out 时使用的第一维 B，必填 |
| `sharedInputWeight` | float | ACLNN 标量参数；sharedInput 合入 out 前使用的乘数 |
| `sharedInputOffset` | int | ACLNN 标量参数；sharedInput 第一行在 out 中的行偏移 |

真实 ACLNN 调用通过预先创建的 `out` Tensor 确定 B，因此没有名为 `output_bs` 的接口参数。Golden 无法从尚未创建的输出反推 B，所以将 `output_bs` 作为必填的输出形状参数。

### Benchmark runner 元数据

以下字段声明在 `proto.yaml` 的 `runner_attrs` 中，并由 runner 在调用 Golden 前消费。它们不参与算子数学计算，也不出现在 Golden、AscendC 或 ACLNN 参数列表中：

| 字段 | 类型 | runner 行为 |
|---|---|---|
| `group_list_values` | list[int] | 按 `groupList` 的 dtype 和 device 构造确定的 `[E]` Tensor，并替换随机输入 |
| `row_index_values` | list[int] | 按 `rowIndex` 的 dtype 和 device 构造确定的 `[M]` Tensor，并替换随机输入 |
| `deterministic` | bool | 调用前设置 `torch.use_deterministic_algorithms`，调用结束后在 `finally` 中恢复原状态 |

runner 完成上述准备后，只向 `grouped_matmul_finalize_routing` 传入 8 个 Tensor 和`output_bs`、`sharedInputWeight`、`sharedInputOffset` 这 3 个 Golden 标量参数。`group_list_values`、`row_index_values` 和 `deterministic` 不得透传给 Golden。

### 输出

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `out` | `[output_bs,7168]` | FLOAT32 | 共享专家和路由专家的最终聚合结果 |

### 规则与约束

- `1 <= M <= 131072`。
- `1 <= E <= 256`。
- Atlas A2/A3 本路径固定 `K=2048`、`N=7168`。
- 所有 Tensor 必须非空。
- `x1.shape == [M,2048]`。
- `x2` 的逻辑 shape 为 `[E,2048,7168]`；真实算子调用前必须通过 `torch_npu.npu_format_cast(x2_npu, 29)` 转换为 FRACTAL_NZ。
- `scale.shape == [E,7168]`。
- `pertokenScaleOptional.shape == logit.shape == rowIndex.shape == [M]`。
- `groupList.shape == [E]`，数值必须非负、单调非递减，最后一个值为 M。
- `1 <= output_bs <= M`。
- `0 <= rowIndex[t] < output_bs`；不要求各输出行出现次数相同。
- `sharedInput.shape == [L,7168]`，`L >= 1`。
- `sharedInputOffset >= 0` 且 `sharedInputOffset + L <= output_bs`。
- `x1` 与 `x2` 不转置，`groupListType` 固定为 0。

### Cases 覆盖范围

| 维度 / 参数 | cases 覆盖 | 备注 |
|---|---:|---|
| `K` | 2048 | 固定满足 A2/A3 路径 |
| `N` | 7168 | 固定满足 A2/A3 路径 |
| `E` | 2、4、8、16、32 | 覆盖少专家、典型 MoE 与多专家 |
| `M` | 17～516 | 覆盖小批量、批量路由和 tiling 边界 |
| `output_bs` | 1～258 | 覆盖单输出行、`output_bs=M` 和常见批量 |
| `M % output_bs` | 0、非 0 | 显式覆盖不能 reshape 为固定 Top-K 的输入 |
| `rowIndex` 分布 | 均匀、非均匀、稀疏、单行冲突、置换 | 覆盖通用 scatter-add |
| `groupList` 分布 | 均衡、长尾、空专家、极端偏斜 | 覆盖专家分组边界 |
| `M/E` | 1.0625～257 | 显式覆盖 tiling 阈值 128、129、256、257 |
| `sharedInputOffset` | 0、3、8、12、16、47、64 | 覆盖起始、部分区间和尾部对齐 |
| `sharedInputWeight` | -0.5～1.25 | 覆盖零、负数、分数、1 和大于 1 |
| `deterministic` | false、true | 覆盖两种运行时模式 |

## 4. 精度要求

采用 [CANN 生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)。

### 浮点输出 `out`

对 FP32 `out` 统计平均相对误差 MERE 和最大相对误差 MARE：

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

`out` 虽然存储为 FP32，但 finalize routing 包含 FP32 scatter/atomic 聚合，
因此按仓库精度规范采用 `intermediate_dtype_inherited`，主导中间 dtype 为 FP32：

| 输出 | dtype | Threshold | 通过条件 |
|---|---|---:|---|
| `out` | FLOAT32 | `2^-13` | `MERE < 2^-13` 且 `MARE < 10 * 2^-13` |

Golden 中的 dtype 操作只保留算子语义所需部分：

- `x1` 和 `x2` 在 PyTorch 中转为 INT32，以表达硬件的 INT8 x INT8、INT32 累加语义；直接执行 INT8 matmul 会产生 INT8 溢出。
- `scale`、`pertokenScaleOptional` 和 `logit` 已由输入契约固定为 FP32，不额外升精度。
- INT32 累加结果与 FP32 scale 相乘后自然进入 FP32 反量化路径。
- `sharedInput` 为 BF16，在乘 `sharedInputWeight` 和合入 out 前转换为 FP32。
- `routed` 和 `out` 均为 FP32。

真实 AscendC 实现使用并行 scatter/atomic add，Golden 使用 `index_add_` 表达相同数学语义。两者的浮点累加顺序可能不同，因此不要求非确定性模式逐位一致，但最终结果仍须满足上表的 MERE 和 MARE 条件。

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
    x1: torch.Tensor,
    x2: torch.Tensor,
    scale: torch.Tensor,
    pertokenScaleOptional: torch.Tensor,
    groupList: torch.Tensor,
    sharedInput: torch.Tensor,
    logit: torch.Tensor,
    rowIndex: torch.Tensor,
    **attrs,
) -> list:
    """从 attrs.group_list_values / row_index_values 重建 groupList、rowIndex 张量。

    cases.yaml 将确定性的 cumsum 分组边界放在 group_list_values 属性、行索引放在
    row_index_values 属性（golden 读取它们），但被测 kernel 只看 groupList / rowIndex
    张量。若不由 get_input 重建，这两个张量会被 value_range 随机生成（可能为负、
    麻烦单调），导致 kernel 与 golden 分组/累加目标不一致。返回值同时替换 golden 与
    AI 算子的输入，确保对比公平。
    """
    gl = attrs.get("group_list_values")
    if gl is not None:
        groupList = torch.tensor(list(gl), dtype=torch.int64, device=x1.device)
    ri = attrs.get("row_index_values")
    if ri is not None:
        rowIndex = torch.tensor(list(ri), dtype=torch.int64, device=x1.device)
    return [x1, x2, scale, pertokenScaleOptional, groupList, sharedInput, logit, rowIndex]


def _cumsum_groups(groupList: torch.Tensor):
    """将 cumsum groupList 转换为各专家的 [start, end) 行区间。"""
    ends = [int(v) for v in groupList.detach().cpu().tolist()]
    starts = [0] + ends[:-1]
    return list(zip(starts, ends))


def grouped_matmul_finalize_routing(
    x1: torch.Tensor,
    x2: torch.Tensor,
    scale: torch.Tensor,
    pertokenScaleOptional: torch.Tensor,
    groupList: torch.Tensor,
    sharedInput: torch.Tensor,
    logit: torch.Tensor,
    rowIndex: torch.Tensor,
    output_bs: int,
    sharedInputWeight: float = 1.0,
    sharedInputOffset: int = 0,
) -> torch.Tensor:
    """AscendC 实现目标及 A2/A3 A8W8 per-token/per-channel 路径 Golden。

    对齐 aclnnGroupedMatmulFinalizeRoutingWeightNzV2 的以下固定路径：
    x1/x2 为 INT8，scale 为 FP32 per-channel，per-token scale 为 FP32，
    sharedInput 为 BF16，rowIndex 为 INT64，输出为 FP32；不含 bias、
    offset 和 antiquant，矩阵不转置，groupListType=0。

    x1 的 M 行是按专家连续排列的路由记录，cumsum groupList 划分各专家的行区间。对第 i 个专家：

        acc_i = MatMul_INT32(x1[start_i:end_i, :], x2[i, :, :])
        routed_i = FP32(acc_i) * scale[i, None, :]
                   * pertokenScaleOptional[start_i:end_i, None]
                   * logit[start_i:end_i, None]
        out[rowIndex[t], :] += routed[t, :]

    sharedInput 会先乘 sharedInputWeight，再从 sharedInputOffset 行开始加入out。
    sharedInputOffset 不参与 rowIndex 的 scatter-add 索引计算。logit由上游提供，本算子不执行 softmax 或归一化。

    输入约定：
        x1: [M, K], INT8
        x2: [E, K, N], INT8 逻辑矩阵；真实算子的 FRACTAL_NZ 转换由适配器负责
        scale: [E, N], FP32
        pertokenScaleOptional: [M], FP32
        groupList: [E], INT64 cumsum
        sharedInput: [L, N], BF16
        logit: [M], FP32
        rowIndex: [M], INT64，取值范围 [0, output_bs)

    输出为 [output_bs, N] FP32，精度路径为 INT8 x INT8 -> INT32 累加 -> FP32 反量化与聚合。
    """
    # x1 已按专家排序，groupList 给出各专家的累积结束位置。
    groups = _cumsum_groups(groupList)
    m = x1.shape[0]
    n = x2.shape[2]
    shared_len = sharedInput.shape[0]
    shared_end = int(sharedInputOffset) + shared_len

    route_scale = logit * pertokenScaleOptional

    routed = torch.empty((m, n), dtype=torch.float32, device=x1.device)
    for expert, (start, end) in enumerate(groups):
        # 显式转为 INT32，以表达 INT8 x INT8 的 INT32 累加。
        acc = torch.matmul(x1[start:end].to(torch.int32), x2[expert].to(torch.int32))
        expert_out = acc * scale[expert].reshape(1, n)
        routed[start:end] = expert_out * route_scale[start:end].reshape(-1, 1)

    # Kernel 对每条路由记录读取 rowIndex，并原子累加到对应输出行。
    out = torch.zeros((output_bs, n), dtype=torch.float32, device=x1.device)
    out[sharedInputOffset:shared_end] = sharedInput.to(torch.float32) * sharedInputWeight
    out.index_add_(0, rowIndex, routed)
    return out


if __name__ == "__main__":
    m, k, n, e = 1, 2048, 7168, 1
    result = grouped_matmul_finalize_routing(
        x1=torch.zeros((m, k), dtype=torch.int8),
        x2=torch.zeros((e, k, n), dtype=torch.int8),
        scale=torch.ones((e, n), dtype=torch.float32),
        pertokenScaleOptional=torch.ones((m,), dtype=torch.float32),
        groupList=torch.tensor([m], dtype=torch.int64),
        sharedInput=torch.zeros((m, n), dtype=torch.bfloat16),
        logit=torch.ones((m,), dtype=torch.float32),
        rowIndex=torch.tensor([0], dtype=torch.int64),
        output_bs=m,
        sharedInputWeight=1.0,
        sharedInputOffset=0
    )
    print(result.shape, result.dtype)
```

`grouped_matmul_finalize_routing` 是唯一 Golden 入口和唯一需要生成 AscendC 实现的函数，不要求逐指令复现 AscendC 的 tiling、流水调度或并行累加顺序。

## 6. 额外信息

### Case 设计

`cases.yaml` 与 `cases.csv` 一一对应，共包含 20 个正向 case：

| case | E | M | output_bs | shared L | offset | M/E | 覆盖重点 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 4 | 32 | 16 | 16 | 0 | 8 | Top-2 均衡基础路径 |
| 2 | 8 | 64 | 32 | 32 | 0 | 8 | Top-2 确定性模式 |
| 3 | 8 | 128 | 64 | 64 | 0 | 16 | Top-2 长尾专家分布 |
| 4 | 8 | 192 | 24 | 24 | 0 | 24 | Top-8 高聚合冲突 |
| 5 | 16 | 192 | 96 | 96 | 0 | 12 | Top-2 与空专家 |
| 6 | 32 | 256 | 128 | 128 | 0 | 8 | 多专家 |
| 7 | 8 | 256 | 64 | 64 | 0 | 32 | Top-4 与 INT8 极值 |
| 8 | 8 | 192 | 96 | 64 | 16 | 24 | 部分 sharedInput 与非零 offset |
| 9 | 4 | 512 | 256 | 256 | 0 | 128 | 默认 tiling 边界 |
| 10 | 4 | 516 | 258 | 258 | 0 | 129 | 进入特殊 tiling 区间 |
| 11 | 2 | 512 | 256 | 256 | 0 | 256 | 特殊 tiling 上边界 |
| 12 | 4 | 40 | 32 | 8 | 12 | 10 | 稀疏目标行与确定性聚合 |
| 13 | 8 | 64 | 32 | 16 | 8 | 8 | 所有路由聚合到同一输出行 |
| 14 | 8 | 65 | 33 | 17 | 16 | 空专家、非均匀路由和尾部 sharedInput |
| 15 | 4 | 48 | 48 | 1 | 47 | 12 | `output_bs=M`、置换索引和负 shared 权重 |
| 16 | 8 | 96 | 1 | 1 | 0 | `output_bs=1`、确定性最大冲突和零 shared 权重 |
| 17 | 16 | 17 | 9 | 3 | 3 | 1.0625 | 多空专家与不等目标行计数 |
| 18 | 2 | 127 | 64 | 32 | 16 | 63.5 | M=127 尾块与非整除 scatter |
| 19 | 2 | 129 | 64 | 64 | 0 | 64.5 | M=129、极端专家负载与确定性 |
| 20 | 4 | 257 | 128 | 64 | 64 | 64.25 | M=257 尾块与非均匀确定性 scatter |

所有 case 固定 `K=2048`、`N=7168`，并使用同一个 `GMMFR_A8W8_IMPL(int64_t, float, true, false)` Kernel 模板。Host tiling 默认采用 `baseM=128、baseN=256、baseK=128`；当 `128 < M/E <= 256` 时采用 `baseM=256、baseN=128、baseK=128`。

输入值域按参数分别设置：

- `x1/x2` 的常规 case 使用 `[-16,16]`，case 7 使用 `[-127,127]` 覆盖 INT8 极值。
- `scale` 与 `pertokenScaleOptional` 使用正值 `[0.0001,0.02]`。
- `logit` 使用 `[0,1]`。
- `sharedInput` 使用 `[-1,1]`。
- `groupList` 与 `rowIndex` 使用每个 case 中显式注入的确定值。

上述值域是测试数据设计，不是算子接口的数值范围约束。

### 实现对齐依据

- `docs/aclnnGroupedMatmulFinalizeRoutingWeightNzV2.md`：WeightNzV2 接口、dtype 与 shape 约束
- `op_kernel/grouped_matmul_finalize_routing.cpp`：A8W8 Kernel 模板映射
- `op_host/grouped_matmul_finalize_routing_base_tiling.cpp`：A8W8 tiling 选择
- `op_api/aclnn_grouped_matmul_finalize_routing.cpp`：shape 校验不要求 `M/output_bs` 整除或 rowIndex 均匀计数
- `op_kernel/grouped_matmul_finalize_routing.h`：按每条 `rowIndex` 执行 scatter/atomic add 的 finalize routing 语义
- `tests/assets/golden.py`：固定 Top-K 测试数据下的参考实现；其 `argsort + reshape + sum` 不扩展为接口约束

### 参考资料

- [CANN aclnnGroupedMatmulFinalizeRoutingWeightNzV2 文档](../../ops-transformer/gmm/grouped_matmul_finalize_routing/docs/aclnnGroupedMatmulFinalizeRoutingWeightNzV2.md)
- [FlatQuant benchmark desc.md 结构参考](../flat_quant_pertoken_int4/desc.md)
