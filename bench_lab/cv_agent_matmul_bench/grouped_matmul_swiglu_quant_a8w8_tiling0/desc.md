# GroupedMatmulSwigluQuant 算子 API 描述

## 1. 算子简介

**算子特征**：
- 难度等级：L4（FusedComposite）

`grouped_matmul_swiglu_quant` 融合 grouped matmul、反量化、SwiGLU 和输出动态 int8 量化。本 benchmark 对齐源码目录 `ops-transformer/gmm/grouped_matmul_swiglu_quant`，选取 `A8W8_tiling_key_0` 路径：AIC 完成 grouped int8 matmul 写 workspace，AIV 执行 `xScale/weightScale` 反量化、SwiGLU、逐 token absmax 和 int8 量化。

该 selected kernel path 是单 kernel C->V 路径。本 benchmark 不覆盖 A8W4 MSD、weight NZ、split workspace 或 v2 跨 workspace 调度路径。

## 2. 算子定义

设 `x` 的形状为 `[M, K]`，`weight` 的形状为 `[E, K, N]`，`weightScale` 的形状为 `[E, N]`，`xScale` 的形状为 `[M]`。`N` 必须为偶数，输出 `y` 的宽度为 `N / 2`。

```text
start = 0
for expert i in [0, E):
    end = groupList[i]
    C = (x[start:end].int32 @ weight[i].int32)
    C = C * xScale[start:end, None] * weightScale[i][None, :]
    C_act, gate = split(C, 2, dim=-1)
    S = swish(C_act) * gate
    yScale[start:end] = max(abs(S), dim=-1) / 127
    y[start:end] = round(S / yScale[:, None]).clamp(-128, 127).to(int8)
    start = end
```

其中 `swish(x) = x / (1 + exp(-x))`。当某一行 `max(abs(S)) == 0` 时，golden 保持 `yScale=0`，对应 `y` 行输出 0。

## 3. 接口规范

benchmark 抽象接口：

```python
grouped_matmul_swiglu_quant(
    x, weight, weightScale, xScale, groupList,
    variant="A8W8_tiling_key_0", tiling_key=0
) -> (y, yScale)
```

参数说明：

| 参数 | 输入/输出 | dtype | shape | 说明 |
|------|-----------|-------|-------|------|
| `x` | 输入 | `INT8` | `[M, K]` | routed token 激活 |
| `weight` | 输入 | `INT8` | `[E, K, N]` | expert 权重，本 benchmark 固定 A8W8 ND 路径 |
| `weightScale` | 输入 | `FLOAT32` | `[E, N]` | per-expert per-channel weight scale |
| `xScale` | 输入 | `FLOAT32` | `[M]` | per-token activation scale |
| `groupList` | 输入 | `INT64` | `[E]` | cumsum token 边界 |
| `y` | 输出 | `INT8` | `[M, N/2]` | SwiGLU 后动态量化输出 |
| `yScale` | 输出 | `FLOAT32` | `[M]` | 每个 token 的输出量化 scale |

## 4. 约束说明

- `variant` 固定为 `A8W8_tiling_key_0`，`tiling_key` 固定为 `0`。
- `groupList` 使用 cumsum 语义，必须非负单调非递减，最后一个值等于 `M`。
- `weight.shape[-1] == N` 必须为偶数。
- 本 benchmark 固定 `weightScale` 为 `[E, N]` 的 FLOAT32 per-channel scale。
- 不覆盖 `bias/offset/weightAssistanceMatrix`、A8W4、weight NZ、split workspace 和跨 kernel v2 调度。

## 5. 精度要求

本算子精度判定遵循 [`../PRECISION_SPEC.md`](../PRECISION_SPEC.md)。通过条件与阈值参数定义在同目录 `proto.yaml` 的 `precision` 节点,以下仅说明本算子特定的取舍。

### 5.1 算子特定说明

- **`y` 阈值归属**:规则 `int8_three_tier`,采用默认参数(fatal=2 / tolerance=1 / bit_exact_ratio=0.99)。
- **`yScale` 阈值归属**:规则 `intermediate_dtype_inherited`。`yScale` 自身 dtype 为 FLOAT32,但其数值由 V 段 SwiGLU 推导:`int8 matmul → fp32 dequant → V 段 swish*gate → ReduceMax → /127`。直接按 FLOAT32 阈值 `2^-13` 会忽略 V 段中间精度上限。
  - **默认假设**:NPU V 段 SwiGLU 使用 FP32 中间,对应阈值 2^-13(`proto.yaml.precision.outputs[yScale].intermediate_dtype = float32`)。
  - **观察项**:若实测 `yScale` MARE 接近 2^-13,说明 V 段实际可能用 FP16 中间;将 `intermediate_dtype` 改为 `float16`(阈值放宽到 2^-10)。
- **空 expert / `yScale = 0`**(空 expert 区间或某行 SwiGLU 后 amax=0):由 SPEC §5 小值特殊处理覆盖,无算子专属逻辑。

## 6. 标准 Golden 代码

`golden.py` 使用 PyTorch float32 完成 grouped matmul、反量化、SwiGLU 和动态量化：

```python
c = torch.matmul(x_i.float(), weight_i.float())
c = c * xScale_i.float().reshape(-1, 1)
c = c * weightScale_i.float().reshape(1, -1)
act, gate = c.chunk(2, dim=-1)
s = (act * torch.sigmoid(act)) * gate
scale = torch.abs(s).amax(dim=-1) / 127.0
q = torch.where(scale[:, None] > 0, s / scale[:, None], 0)
q = torch.round(q).clamp(-128, 127).to(torch.int8)
```

## 7. 额外信息

### 测试资料对应关系

- `docs/aclnnGroupedMatmulSwigluQuant.md`：描述 A8W8 公式、groupList cumsum 语义、SwiGLU 和输出动态量化。
- `op_kernel/grouped_matmul_swiglu_quant.cpp`：`tiling_key=0` 调用 `GMM_CV_SPLIT_IMP(GMMSwigluCompute, ...)`。
- `op_kernel/grouped_matmul_swiglu_quant.h`：vector 侧执行 `SwiGLU`、`ReduceMax`、`quantScale=max(abs(S))/127` 和 int8 cast。

### 本 benchmark case 设计

`cases.yaml` 当前包含 20 个正向 case。case 1-6 是小 shape/边界烟测，覆盖单 expert、多 expert、空 expert、tail M、非均匀 group 和 many experts；case 7-14 是标准档位 LLM prefill baseline，覆盖 `M=256/512/1024/2048`、`K=1024/2048/4096`、`N=2048/4096` 和 `E=2/4`；case 15-20 引入非对齐大 shape，覆盖非 256 对齐的 `M=640/1000/3500`、非标准 `K=768/1152/1536/2304`、非标准 `N=1536/2304/3072`、`E=4/8/16`、ragged group 和 multiple empty experts。所有 case 固定 A8W8 `tiling_key=0`，最大单维限制在 4096，便于本地测试。

## 标准 Golden 代码

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
    weightScale: torch.Tensor,
    xScale: torch.Tensor,
    groupList: torch.Tensor,
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
    return [x, weight, weightScale, xScale, groupList]


def grouped_matmul_swiglu_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    weightScale: torch.Tensor,
    xScale: torch.Tensor,
    groupList: torch.Tensor,
    variant: str = "A8W8_tiling_key_0",
    group_list_values=None,
    tiling_key: int = 0,
):
    """Torch golden for grouped_matmul_swiglu_quant A8W8 tiling_key=0."""
    if variant != "A8W8_tiling_key_0" or tiling_key != 0:
        raise ValueError("This benchmark fixes grouped_matmul_swiglu_quant A8W8 tiling_key=0")
    if x.dim() != 2:
        raise ValueError(f"x expects 2D [M,K], got {list(x.shape)}")
    if weight.dim() != 3:
        raise ValueError(f"weight expects 3D [E,K,N], got {list(weight.shape)}")

    m, k = x.shape
    expert_num, wk, n = weight.shape
    if wk != k:
        raise ValueError(f"weight K ({wk}) must match x K ({k})")
    if n % 2 != 0:
        raise ValueError("weight last dimension N must be even for SwiGLU split")
    if weightScale.shape != (expert_num, n):
        raise ValueError(f"weightScale expects shape [{expert_num}, {n}], got {list(weightScale.shape)}")
    if xScale.numel() != m:
        raise ValueError(f"xScale length ({xScale.numel()}) must match M ({m})")

    hidden = n // 2
    groups = _cumsum_group_list(groupList, m, expert_num, group_list_values)
    y = torch.zeros(m, hidden, dtype=torch.int8, device=x.device)
    y_scale = torch.zeros(m, dtype=torch.float32, device=x.device)

    start = 0
    for expert_id, end in enumerate(groups):
        if end == start:
            continue
        x_i = x[start:end].to(torch.float32)
        w_i = weight[expert_id].to(torch.float32)
        c = torch.matmul(x_i, w_i)
        c = c * xScale[start:end].to(torch.float32).reshape(-1, 1)
        c = c * weightScale[expert_id].to(torch.float32).reshape(1, n)

        act, gate = c.chunk(2, dim=-1)
        s = (act * torch.sigmoid(act)) * gate

        abs_max = torch.abs(s).amax(dim=-1)
        scale = abs_max / 127.0
        safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        normalized = s / safe_scale.reshape(-1, 1)
        normalized = torch.where(scale.reshape(-1, 1) > 0, normalized, torch.zeros_like(s))
        q = torch.round(normalized).clamp(-128, 127).to(torch.int8)
        y[start:end] = q
        y_scale[start:end] = scale
        start = end

    return y, y_scale


def _cumsum_group_list(group_list, total_m: int, group_num: int, group_list_values):
    if group_list_values is not None:
        values = torch.tensor(group_list_values, dtype=torch.int64)
    else:
        values = group_list.to(torch.int64).flatten()
    if values.numel() != group_num:
        raise ValueError(f"groupList length ({values.numel()}) must match expert count ({group_num})")
    if bool(torch.any(values < 0)):
        raise ValueError("groupList values must be non-negative")
    if bool(torch.any(values[1:] < values[:-1])):
        raise ValueError("groupList must be non-decreasing cumsum")
    if int(values[-1]) != total_m:
        raise ValueError(f"groupList last value ({int(values[-1])}) must equal M ({total_m})")
    return [int(v) for v in values.tolist()]
```
