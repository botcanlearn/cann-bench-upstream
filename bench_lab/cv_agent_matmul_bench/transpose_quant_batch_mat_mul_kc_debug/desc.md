# TransposeQuantBatchMatMul 算子 API 描述

## 1. 算子简介

`transpose_quant_batch_mat_mul` 对齐源码目录 `ops-nn/matmul/transpose_quant_batch_mat_mul`。本 benchmark 选取 K-C 量化路径：先按 `permX1/permX2` 调整 batch matmul 输入视图，C 侧完成 INT8 batch matmul，V 侧执行 per-token/per-channel scale、bias 和可选 `permY`，属于 C->V kernel flow。

## 2. 算子定义

```text
A = permute(x1, permX1)
B = permute(x2, permX2)
Y = (A @ B) * x1Scale[..., :, None] * x2Scale[..., None, :] + bias[..., None, :]
out = permute(Y, permY)
```

## 3. 接口规范

```python
transpose_quant_batch_mat_mul(x1, x2, x1Scale, x2Scale, bias, permX1, permX2, permY, groupSize=0) -> out
```

| 参数 | 输入/输出 | dtype | shape | 说明 |
|------|-----------|-------|-------|------|
| `x1` | 输入 | `INT8` | `[B,M,K]` 或 `[B,K,M]` | 左矩阵，按 `permX1` 解释 |
| `x2` | 输入 | `INT8` | `[B,K,N]` 或 `[B,N,K]` | 右矩阵，按 `permX2` 解释 |
| `x1Scale` | 输入 | `FLOAT32` | `[B,M]` | per-token scale |
| `x2Scale` | 输入 | `FLOAT32` | `[B,N]` | per-channel scale |
| `bias` | 输入 | `FLOAT32` | `[B,N]` | 浮点 bias |
| `out` | 输出 | `FLOAT32` | `[B,M,N]` 或 `[B,N,M]` | 输出矩阵 |

## 4. 约束说明

- 固定 K-C 路径，`K=512`、`N=128`，`groupSize=0`。
- 覆盖 `permX1/permX2/permY` 的常见 `[0,1,2]` 与 `[0,2,1]` 组合。
- 目录名中的 `_kc` 用于明确本 benchmark 固定 K-C 量化路径；MX 量化和 FP8 细节不纳入本目录。

## 5. 精度要求

本算子精度判定遵循 [`../PRECISION_SPEC.md`](../PRECISION_SPEC.md)。通过条件与阈值参数定义在同目录 `proto.yaml` 的 `precision` 节点,以下仅说明本算子特定的取舍。

### 5.1 算子特定说明

- **`out` 阈值归属**:规则 `output_dtype`。输出为 float32 golden，重点验证 transpose 视图与 scale/bias 广播。

## 6. 标准 Golden 代码

`golden.py` 使用 `permute` 构造逻辑输入，再执行 batch matmul、scale/bias 和输出置换。

## 7. 额外信息

### 测试资料对应关系

- `docs/aclnnTransposeQuantBatchMatMul.md`：K-C/MX 约束、`permX1/permX2/permY` 和 `groupSize` 说明。
- `op_kernel/arch35/transpose_quant_batch_mat_mul.cpp`：arch35 kernel 入口。

### 本 benchmark case 设计

`cases.yaml` 当前包含 20 个正向 case，覆盖 batch、不同 `M`、三类 perm 组合和输出转置。

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


def transpose_quant_batch_mat_mul(
    x1: torch.Tensor,
    x2: torch.Tensor,
    x1Scale: torch.Tensor,
    x2Scale: torch.Tensor,
    bias: torch.Tensor,
    permX1=(0, 1, 2),
    permX2=(0, 1, 2),
    permY=(0, 1, 2),
    groupSize: int = 0,
    batchSplitFactor: int = 1,
    y_dtype: str = "float32",
) -> torch.Tensor:
    """Torch golden for transpose_quant_batch_mat_mul K-C path."""
    a = x1.permute(*permX1).to(torch.float32)
    b = x2.permute(*permX2).to(torch.float32)
    if a.dim() != 3 or b.dim() != 3:
        raise ValueError("This benchmark fixes 3D batched inputs")
    batch, m, k = a.shape
    batch2, k2, n = b.shape
    if batch != batch2 or k != k2:
        raise ValueError("shape mismatch after permute")
    # int8 matmul accumulates int32 in the cube PE, but the arch35 kernel's
    # cT = MatmulType<VECIN, ND_ALIGN, l0cDtype=float> (transpose_quant_batch_mat_mul_
    # asw_kernel_advanced.h) makes GetTensorC(l0cOutUb_, 0, true) in MMCompute() land the
    # accumulator in UB already cast to float32 (framework fixpipe), before VFDoDequant
    # ever reads it. That int32->fp32 cast and the plain fp32 matmul below are both
    # bit-exact here (K=512 => |acc| <= 512*127*127 < 2^24 stays exactly representable
    # in fp32), so torch.matmul in fp32 reproduces the hardware accumulation regardless
    # of summation order.
    y = torch.matmul(a, b)
    # Dequant order MUST be x2Scale (per-channel N) then x1Scale (per-token M):
    # VFDoDequant in ..._asw_kernel_advanced.h does mul(scale) then mul(perTokenScale);
    # repo tests/assets golden _kc_matmul does the same. Do NOT pre-combine the scales.
    y = y * x2Scale.to(torch.float32).reshape(batch, 1, n)
    y = y * x1Scale.to(torch.float32).reshape(batch, m, 1)
    # bias added last, after both scales (design.md epilogue: ((mm*x2Scale)*x1Scale)+bias).
    y = y + bias.to(torch.float32).reshape(batch, 1, n)
    return y.permute(*permY)
```
