# QuantGroupedMatmulDequant 算子 API 描述

## 1. 算子简介

`quant_grouped_matmul_dequant` 对齐主线算子 `aclnnQuantGroupedMatmulDequant`（`ops-transformer/gmm/quant_grouped_matmul_dequant`）的 **公式 4.1 + 动态 per-token 量化场景**。V 侧对浮点 `x` 按 token 动态计算 `xScale` 并量化，C 侧执行 grouped INT8 matmul，V 侧按 `weightScale` 和内部 `xScale` 反量化，属于 V->C->V kernel flow。主线 `transposeWeight` 强制 true、x 仅 FLOAT16（对齐主线）、`out` = x dtype；`bias` 文档公式 L47 含（scale 前加）、文档 L166 当前实现要求 null、case 不带 bias、golden 保留计算。

## 2. 算子定义

```text
xScale = row_max(abs(x)) / 127
x_quantized = round(x / xScale)        # 截断到 [-128,127]
for group i:
    out[start:end] = (x_quantized[start:end] @ weight[i]) * weightScale[i] * xScale[start:end]
```

`bias` 在文档公式 L47 中 scale 前加（`out=(xq@w+bias)*scale`）；文档 L166 当前实现要求 bias=null，故 case 不带 bias，golden 保留 bias 计算逻辑（默认 None）。

## 3. 接口规范

```python
quant_grouped_matmul_dequant(x, quantized_weight, weightScale, groupList, quant_mode="pertoken", xScaleOptional=None) -> out
```

| 参数 | 输入/输出 | dtype | shape | 说明 |
|------|-----------|-------|-------|------|
| `x` | 输入 | `FLOAT16/BFLOAT16` | `[M,K]` | 未量化激活（主线仅 FLOAT16） |
| `quantized_weight` | 输入 | `INT8` | `[G,N,K]` | 量化权重（transposeWeight=true） |
| `weightScale` | 输入 | `FLOAT32` | `[G,N]` | per-channel 权重 scale |
| `groupList` | 输入 | `INT64` | `[G]` | cumsum 分组边界 |
| `out` | 输出 | `= x dtype` | `[M,N]` | 反量化输出 |

## 4. 约束说明

- 固定 `quant_mode=pertoken`、`xScaleOptional=None`、`transposeWeight=true`（对齐主线，weight 按 `[G,N,K]` 解释）、`groupListType=0`。
- `bias`：文档公式 L47 含（scale 前加）；文档 L166 当前实现要求 null，case 不带 bias；golden 保留 bias 计算逻辑（bias=None，传入则按公式加）。
- `groupList` 允许空 group，最后一个值等于 `M`。
- 主线 N、K 需 16 整数倍；主线 x 仅 FLOAT16，本 benchmark 保留 fp16/bf16 覆盖、out = x dtype。
- `smoothScale` 和 `weightScale=INT64 + xScale=FP16 + pertensor` 静态路径不纳入本目录。

## 5. 精度要求

本算子精度判定遵循 [`benchmark_spec.md` §4.4](../../../docs/spec/benchmark_spec.md)。通过条件与阈值参数定义在同目录 `proto.yaml` 的 `precision` 节点,以下仅说明本算子特定的取舍。

### 5.1 算子特定说明

- **`out` 阈值归属**:规则 `input_dtype_inherited`。输入为 FP16/BF16，输出反量化误差受动态量化 round-trip 影响。

## 6. 标准 Golden 代码

`golden.py` 内部计算 per-token `xScale`，再对每个 group 执行 INT8 matmul 与反量化（无 bias）。详见同目录 `golden.py`。

## 7. 额外信息

### 测试资料对应关系

- `docs/aclnnQuantGroupedMatmulDequant.md`：动态 per-token 量化、`xScaleOptional` 为空时的公式 4.1。
- `op_kernel/quant_grouped_matmul_dequant_normal.h`：普通 grouped dequant 路径。

### 本 benchmark case 设计

`cases.yaml` 当前包含 20 个正向 case，覆盖多 group、空 group、非均匀 group、BF16/FP16 和不同 K/N。

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


# ===== 规格对齐说明(对齐主线 aclnnQuantGroupedMatmulDequant 的公式4.1+动态pertoken场景) =====
# 主线算子: aclnnQuantGroupedMatmulDequant (非 WeightNZ 变体)
#   实现: ops-transformer/gmm/quant_grouped_matmul_dequant
# 对应场景: 公式 4.1(weightScale=FLOAT32) + xQuantMode=pertoken + xScaleOptional=null(动态量化)
# 主线公式:
#   xScale[m]   = rowmax(|x[m,:]|) / 127
#   xq[m,:]     = round_half_to_even(x / xScale), 截断到 [-128,127]
#   out[m,n]    = (xq[start:end] @ weight[i] + bias) * weightScale[i,n] * xScale[m]   (bias 在 scale 前加, 文档 L47)
#   出处: docs/aclnnQuantGroupedMatmulDequant.md (公式2/3/4.1);
#         op_host/quant_grouped_matmul_dequant_tiling.cpp (L138-141 bias不支持, L143-150 动态强制pertoken, L901 transposeWeight只支持true);
#         op_api/aclnn_quant_grouped_matmul_dequant.cpp (L80-83, L191-193);
#         op_kernel/quant_grouped_matmul_dequant_phase_x.h / phase_mm.h
# 主线 dtype/shape 规格:
#   x        : FLOAT16, 2D (M,K)            [主线仅 FLOAT16]
#   quantized_weight: INT8, 3D (G,N,K)     [transposeWeight 强制 true]
#   weightScale: FLOAT32, 2D (G,N)
#   groupList: INT64, 1D (G,), cumsum 边界(末值=M)
#   bias     : 文档公式 L47 含(scale 前加); 文档 L166 当前实现要求 null; case 不带 bias; golden 保留计算
#   xScaleOptional: 本场景 nullptr(动态量化, 强制 pertoken)
#   out      : FLOAT16(= x dtype), 2D (M,N)
# 关键约束: N、K 16 整数倍; transposeWeight=true; bias/xOffset 必须空; 主线仅 310P.
# 本 golden 与主线的对齐说明:
#   - bias: 文档 L47 公式含(scale 前加); 文档 L166 当前实现要求 null; case 不带 bias(bias=None);
#     golden 保留 bias 计算逻辑(if bias)对齐公式, 将来实现放开即生效。
#   - transposeWeight 固定 true: weight 输入按 [G,N,K] 解释, 内部 transpose 成 [G,K,N] 使用。
#   - out cast 到 x.dtype(对齐主线 out = x dtype; 主线 x 仅 FLOAT16, 本 benchmark 保留 fp16/bf16 覆盖)。
#   - 动态 per-token 量化保留(主线有): amax/127 + round_half_to_even + 截断 [-128,127](对齐 INT8 有符号范围)。
#   - clamp_min(eps) 防 0 除为主线无的防御(注明); 行归约用 fp32(优于主线 fp16, 注明)。
#   - 计算用 torch.float32 表达等价数学, 不做硬件 dtype 物理转换。
# ===========================================================================================


def _groups(groupList: torch.Tensor, groupListType: int):
    values = [int(v) for v in groupList.detach().cpu().tolist()]
    if groupListType == 0:
        starts = [0] + values[:-1]
        ends = values
    elif groupListType == 1:
        starts, ends, cur = [], [], 0
        for count in values:
            starts.append(cur)
            cur += count
            ends.append(cur)
    else:
        raise ValueError("groupListType must be 0 or 1")
    return list(zip(starts, ends))


def get_input(
    x: torch.Tensor,
    quantized_weight: torch.Tensor,
    weightScale: torch.Tensor,
    groupList: torch.Tensor,
    **attrs,
) -> list:
    """从 attrs.group_list_values 重建确定性的 groupList 张量。

    cases.yaml 将 cumsum 分组边界放在 group_list_values 属性里（golden 读取它），
    但被测 kernel 只看 groupList 张量。若不由 get_input 重建，groupList 会被
    value_range 随机生成（可能为负、非单调），导致 kernel 与 golden 分组不一致。
    返回值同时替换 golden 与 AI 算子的输入，确保对比公平。
    bias 不在 proto inputs 中（case 固定 bias=None），故不参与 get_input。
    """
    gl = attrs.get("group_list_values")
    if gl is not None:
        groupList = torch.tensor(list(gl), dtype=torch.int64, device=x.device)
    return [x, quantized_weight, weightScale, groupList]


def quant_grouped_matmul_dequant(
    x: torch.Tensor,
    quantized_weight: torch.Tensor,
    weightScale: torch.Tensor,
    groupList: torch.Tensor,
    bias: torch.Tensor = None,
    quant_mode: str = "pertoken",
    xScaleOptional=None,
    transposeWeight: bool = True,
    groupListType: int = 0,
    group_list_values=None,
) -> torch.Tensor:
    """Torch golden for quant_grouped_matmul_dequant, aligned to aclnnQuantGroupedMatmulDequant (公式4.1+动态pertoken).

    动态 per-token 量化 -> INT8 grouped matmul -> weightScale*xScale 反量化(bias 恒 0), out = x dtype.
    规格详见模块顶部「规格对齐说明」。
    """
    if quant_mode != "pertoken" or xScaleOptional is not None:
        raise ValueError("This benchmark fixes dynamic per-token path with xScaleOptional=None")
    if group_list_values is not None:
        groupList = torch.tensor(group_list_values, dtype=torch.int64, device=x.device)
    # transposeWeight 固定 true(对齐主线): weight 输入 [G,N,K] -> 内部 [G,K,N]
    if transposeWeight:
        quantized_weight = quantized_weight.transpose(-2, -1)
    groups = _groups(groupList, groupListType)
    g, k, n = quantized_weight.shape
    if len(groups) != g or x.shape[1] != k:
        raise ValueError("shape mismatch")
    # 主线硬约束(文档 L347/L289): weight 的 N、K 需 16 整数倍
    if n % 16 != 0 or k % 16 != 0:
        raise ValueError(f"N,K must align to 16 (docs L347/L289), got N={n} K={k}")
    eps = torch.finfo(torch.float32).tiny
    # 动态 per-token 量化(主线有): amax/127, round_half_to_even, 截断 [-128,127]
    x_scale = x.to(torch.float32).abs().amax(dim=1).clamp_min(eps) / 127.0
    xq = torch.round(x.to(torch.float32) / x_scale.reshape(-1, 1)).clamp(-128, 127)
    out = torch.zeros(x.shape[0], n, dtype=torch.float32, device=x.device)
    for idx, (start, end) in enumerate(groups):
        if end <= start:
            continue
        # 公式 4.1: out = (xq@weight + bias) * weightScale * xScale
        # 文档 L47 公式含 bias(scale 前加); 文档 L166 当前实现要求 bias=null; case 不带 bias;
        # golden 保留 bias 计算逻辑(if bias)对齐公式, 将来实现放开即生效。
        acc = xq[start:end, :] @ quantized_weight[idx].to(torch.float32)
        if bias is not None:
            acc = acc + bias[idx].to(torch.float32).reshape(1, n)
        out[start:end, :] = acc * weightScale[idx].to(torch.float32).reshape(1, n) * x_scale[start:end].reshape(-1, 1)
    # out dtype = x dtype(对齐主线 out = x dtype)
    return out.to(x.dtype)
```
