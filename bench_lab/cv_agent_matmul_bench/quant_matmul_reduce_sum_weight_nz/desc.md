# QuantMatmulReduceSum 算子 API 描述

## 1. 算子简介

`quant_matmul_reduce_sum` 完成一组 batch int8 矩阵乘，经过 scale 反量化后在 batch 维求和。本 benchmark 对齐源码目录 `ops-nn/matmul/quant_matmul_reduce_sum` 中的 `aclnnQuantMatmulReduceSumWeightNz` 语义：`x1` 为 ND int8，`x2` 为 NZ int8 权重，AIC 完成分块 matmul，AIV 做 `x2Scale/x1Scale` 反量化并通过 batch 维累加写回。

该路径是 C->V 算子：cube 侧输出 int32 matmul 中间结果，vector 侧完成反量化与 reduce-sum。

## 2. 算子定义

设 `x1` 的形状为 `[B, M, K]`，`x2` 的逻辑原始形状为 `[B, K, N]`，物理 NZ 形状为 `[B, ceil(N/32), ceil(K/16), 16, 32]`。

$$
out = \sum_{b=0}^{B-1} (x1_b @ x2_b) \odot x1Scale_b[:, None] \odot x2Scale[None, :]
$$

其中 `x1Scale` 的形状为 `[B, M]`，`x2Scale` 的形状为 `[N]`，输出 `out` 的形状为 `[M, N]`。

## 3. 接口规范

benchmark 抽象接口：

```python
quant_matmul_reduce_sum(
    x1, x2, x1Scale, x2Scale,
    dims=[0], keep_dims=False, x2_format="NZ"
) -> out
```

参数说明：

| 参数 | 输入/输出 | dtype | shape | 说明 |
|------|-----------|-------|-------|------|
| `x1` | 输入 | `INT8` | `[B, M, K]` | batch 量化激活 |
| `x2` | 输入 | `INT8` | `[B, ceil(N/32), ceil(K/16), 16, 32]` | NZ 权重，逻辑 shape 为 `[B, K, N]` |
| `x1Scale` | 输入 | `FLOAT32` | `[B, M]` | per-token 激活 scale |
| `x2Scale` | 输入 | `BFLOAT16` | `[N]` | per-channel 权重 scale |
| `dims` | 输入 | `int[]` | `[1]` | 本 benchmark 固定 `[0]` |
| `keep_dims` | 输入 | `bool` | 标量 | 本 benchmark 固定 `False` |
| `out` | 输出 | `BFLOAT16` | `[M, N]` | batch 维归约后的反量化 matmul |

## 4. 约束说明

- `dims` 固定为 `[0]`，只覆盖 batch 维求和。
- `keep_dims` 固定为 `False`。
- `x2_format` 固定为 `NZ`，不覆盖 ND 权重路径。
- 预留参数 `yScale/x1Offset/x2Offset/yOffset/bias` 不在本 benchmark 范围内。
- `x2` NZ 物理形状必须满足 `k1 == ceil(K / 16)`、`n1 == ceil(N / 32)`、尾部 padding 只在还原逻辑 ND 时裁掉。

## 5. 精度要求

本算子精度判定遵循 [`../PRECISION_SPEC.md`](../PRECISION_SPEC.md)。通过条件与阈值参数定义在同目录 `proto.yaml` 的 `precision` 节点,以下仅说明本算子特定的取舍。

### 5.1 算子特定说明

- **`out` 阈值归属**:规则 `output_dtype`,固定 BFLOAT16 → 阈值 2^-7。
- **大 batch 累加风险**:V 段 batch 维 reduce 若使用 fp32 中间再 cast 到 bf16,本阈值合理;若 NPU 实现采用 bf16 累加 + 大 B(≥16),MARE 可能逼近阈值。**实测若超阈值**,需把 `proto.yaml.precision.outputs[out].threshold_rule` 改为 `intermediate_dtype_inherited` 并补 `intermediate_dtype: bfloat16`(阈值仍 2^-7 但语义更准),或考虑放宽到 2^-6。

## 6. 标准 Golden 代码

`golden.py` 先将 NZ 权重还原为逻辑 ND `[B, K, N]`，再执行 matmul、scale 和 batch 维求和：

```python
x2_nd = x2.permute(0, 2, 3, 1, 4).contiguous().reshape(B, K1 * 16, N1 * 32)
x2_nd = x2_nd[:, :K, :N]
mm = torch.matmul(x1.float(), x2_nd.float())
mm = mm * x1Scale.float().reshape(B, M, 1)
mm = mm * x2Scale.float().reshape(1, 1, N)
out = mm.sum(dim=0).to(torch.bfloat16)
```

## 7. 额外信息

### 测试资料对应关系

- `docs/aclnnQuantMatmulReduceSumWeightNz.md`：描述 `x1/x2/x1Scale/x2Scale`、NZ 形状和 batch 维求和公式。
- `op_kernel/quant_matmul_reduce_sum_mixcore.h`：vector 侧完成 `x2Scale` 反量化、`x1Scale` 广播乘法和 atomic 累加。
- `tests/ut/op_kernel/test_quant_matmul_reduce_sum.cpp`：包含 `B=8, M=32, K=64, N=32` 的 kernel 基础样例。

### 本 benchmark case 设计

`cases.yaml` 当前包含 20 个正向 case，覆盖不同 `B/M/K/N`、NZ padding、`K/N` 非 16/32 整除、单 batch、tail M 和中等规模 batch 归约。所有 case 都固定 `dims=[0]`、`keep_dims=false`、`x2_format=NZ`。

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


def quant_matmul_reduce_sum(
    x1: torch.Tensor,
    x2: torch.Tensor,
    x1Scale: torch.Tensor,
    x2Scale: torch.Tensor,
    dims=(0,),
    keep_dims: bool = False,
    x2_format: str = "NZ",
) -> torch.Tensor:
    """Torch golden for quant_matmul_reduce_sum with x2 in NZ layout."""
    if tuple(dims) != (0,):
        raise ValueError("This benchmark fixes dims=[0]")
    if keep_dims:
        raise ValueError("This benchmark fixes keep_dims=False")
    if str(x2_format).upper() != "NZ":
        raise ValueError("This benchmark fixes x2_format=NZ")
    if x1.dim() != 3:
        raise ValueError(f"x1 expects 3D [B,M,K], got {list(x1.shape)}")
    if x2.dim() != 5:
        raise ValueError(f"x2 expects 5D NZ [B,N1,K1,16,32], got {list(x2.shape)}")

    b, m, k = x1.shape
    n = x2Scale.numel()
    if x1Scale.shape != (b, m):
        raise ValueError(f"x1Scale expects shape [{b}, {m}], got {list(x1Scale.shape)}")

    x2_nd = _nz_weight_to_nd(x2, b, k, n)
    # cType=int32 (kernel .cpp): int8xint8 matmul accumulates exactly in int32.
    mm = torch.matmul(x1.to(torch.int32), x2_nd.to(torch.int32)).to(torch.float32)
    # AscendDequant applies bf16 x2Scale FIRST (mixcore.h: x2ScaleGm is bfloat16_t),
    # then per-token x1Scale (fp32) is multiplied; both stages stay fp32.
    mm = mm * x2Scale.to(torch.bfloat16).to(torch.float32).reshape(1, 1, n)
    mm = mm * x1Scale.to(torch.float32).reshape(b, m, 1)
    # Kernel Casts each batch to bf16 (RoundMode::CAST_RINT) then AtomicAdd<bf16>
    # into a zero-init bf16 output, so batch reduce accumulates in bfloat16.
    out = mm[0].to(torch.bfloat16)
    for bi in range(1, b):
        out = (out.to(torch.float32) + mm[bi].to(torch.bfloat16).to(torch.float32)).to(torch.bfloat16)
    return out


def _nz_weight_to_nd(x2: torch.Tensor, batch: int, k: int, n: int) -> torch.Tensor:
    b, n1, k1, k0, n0 = x2.shape
    if b != batch:
        raise ValueError(f"x2 batch ({b}) must match x1 batch ({batch})")
    if k0 != 16 or n0 != 32:
        raise ValueError(f"NZ x2 expects k0=16,n0=32, got k0={k0}, n0={n0}")
    if k1 != (k + 15) // 16:
        raise ValueError(f"x2 K1 ({k1}) must equal ceil(K/16) for K={k}")
    if n1 != (n + 31) // 32:
        raise ValueError(f"x2 N1 ({n1}) must equal ceil(N/32) for N={n}")
    nd = x2.permute(0, 2, 3, 1, 4).contiguous().reshape(b, k1 * k0, n1 * n0)
    return nd[:, :k, :n]
```
