# QuantBatchMatmulV4 算子 API 描述

## 1. 算子简介

`quant_batch_matmul_v4` 对齐主线算子 `aclnnQuantMatmulV5`（`ops-nn/matmul/quant_batch_matmul_v4`）的 **G-B per-group 静态量化场景**。消费**已量化 INT8** 的 `x1/x2` 与**预存 FLOAT32** 的 `x1Scale/x2Scale`，沿 K 按 128 分组 block matmul 累加反量化，加 bias，输出 bf16，属于 C->V kernel flow（无在线动态量化）。主线 `transposeX2=true`、`N%256==0`、`K%512==0`。

## 2. 算子定义

```text
gsK = 128
for each K group j (block j*128:(j+1)*128):
    block = x1[:, Kj] @ x2[Kj, :]            # INT8×INT8 block matmul (x2 已转置 [K,N])
    out += block * x1Scale[m, j] * x2Scale[floor(n/128), j]
out += bias                                   # FLOAT32 bias
out -> bfloat16                               # cast
```

固定 `group_size=[1,128,128]`；`x1Scale`/`x2Scale` 为预存静态 scale（非在线计算）。

## 3. 接口规范

```python
quant_batch_matmul_v4(x1, x2, x1_scale, x2_scale, bias, variant="GB_STATIC_PERGROUP", group_size=(1,128,128)) -> out
```

| 参数 | 输入/输出 | dtype | shape | 说明 |
|------|-----------|-------|-------|------|
| `x1` | 输入 | `INT8` | `[M,K]` | 已量化左矩阵（transposeX1=false） |
| `x2` | 输入 | `INT8` | `[N,K]` | 已量化右矩阵（transposeX2=true） |
| `x1_scale` | 输入 | `FLOAT32` | `[M,⌈K/128⌉]` | pergroup scale 沿 K |
| `x2_scale` | 输入 | `FLOAT32` | `[⌈N/128⌉,⌈K/128⌉]` | perblock scale（文档：N-group 在前） |
| `bias` | 输入 | `FLOAT32` | `[N]` | 浮点 bias |
| `out` | 输出 | `BFLOAT16` | `[M,N]` | 反量化累加输出 |

## 4. 约束说明

- 固定 `variant=GB_STATIC_PERGROUP`、`transpose_x1=false`、`transpose_x2=true`、`group_size=[1,128,128]`。
- 输入为**预存静态已量化数据**（INT8 x1/x2 + 预存 FLOAT32 scale），无在线动态量化。
- 主线 shape 约束：`K%512==0`（即 4×128）、`N%256==0`。

## 5. 精度要求

本算子精度判定遵循 [`../PRECISION_SPEC.md`](../PRECISION_SPEC.md)。通过条件与阈值参数定义在同目录 `proto.yaml` 的 `precision` 节点,以下仅说明本算子特定的取舍。

### 5.1 算子特定说明

- **`out` 阈值归属**:规则 `output_dtype`。输出 BFLOAT16，cast RINT(round-to-nearest-even)。

## 6. 标准 Golden 代码

`golden.py` 按 `groupSizeK=128` 分块 INT8 block matmul，乘预存 scale 累加，加 bias，cast bf16。详见同目录 `golden.py`。

## 7. 额外信息

### 测试资料对应关系

- `docs/aclnnQuantMatmulV5.md`：G-B 公式（L99-107/L204-210）、A2 约束（L655-664/L869-870）、dtype 表。
- `tests/assets/golden.py::_compute_per_tile_int8`：主线 G-B golden 参考。
- `op_kernel/quant_batch_matmul_v4_perblock.h`：perblock kernel。

### 本 benchmark case 设计

`cases.yaml` 当前包含 20 个正向 case，shape 满足主线 `K%512==0、N%256==0`（K∈{512,1024}、N∈{256,512,768,1024}），覆盖不同 M。

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


# ===== 规格对齐说明(对齐主线 aclnnQuantMatmulV5 的 G-B pergroup 静态量化场景) =====
# 主线算子: aclnnQuantMatmulV5 (graph op: QuantBatchMatmulV5, 实现 quant_batch_matmul_v4)
# 对应场景: G-B 量化 (pergroup-perblock, group=(1,128,128)), 静态已量化输入
# 主线公式:
#   gsK = 128
#   out[m,n] = bias[n] + sum_{j=0}^{ceil(K/128)-1} (x1_blk[m,Kj] @ x2_blk[Kj,n]) * x1Scale[m,j] * x2Scale[floor(n/128), j]
#   (x1/x2 已 INT8 量化, scale 预存 FLOAT32, 无在线 amax/round/clamp)
#   出处: docs/aclnnQuantMatmulV5.md (G-B 公式 L99-107/L204-210, A2 约束 L655-664/L869-870, dtype 表 L653);
#         tests/assets/golden.py::_compute_per_tile_int8 (L323-453);
#         op_kernel/quant_batch_matmul_v4_perblock.h; tests/.../gen_data.py (L30-44 scale 随机预生成)
# 主线 dtype/shape 规格:
#   x1        : INT8, [M,K]   (transposeX1=false, 已量化)
#   x2        : INT8, [N,K]   (transposeX2=true, 已量化; 逻辑 [K,N] 参与计算)
#   x1_scale  : FLOAT32, [M, ceil(K/128)]            (pergroup 沿 K)
#   x2_scale  : FLOAT32, [ceil(N/128), ceil(K/128)]  (perblock; 文档 shape: N-group 在前; golden 内部 transpose)
#   bias      : FLOAT32, [N]
#   out       : BFLOAT16, [M,N]   (cast RINT)
#   groupSize : [1,128,128] (编码值 4303356032 = gsK | gsN<<16 | gsM<<32)
# 关键约束: A2 上 K%512==0(4*128)、N%256==0; 32B 对齐; 非空.
# 本 golden 与主线的对齐说明:
#   - 删除在线 amax/127/round/clamp 动态量化(对齐主线静态契约: 消费已量化 x1/x2 + 预存 scale)。
#   - 沿 K 按 128 分组: INT8 block matmul(int32 累加, fp32 表达) * x1Scale*x2Scale 累加 + bias, cast bf16。
#   - x2 按主线 transposeX2=true 解释([N,K] -> [K,N] 参与)。
#   - 计算用 torch.float32 表达等价数学, 不做 INT8 物理 dtype 存储转换。
# ===========================================================================================


def quant_batch_matmul_v4(
    x1: torch.Tensor,
    x2: torch.Tensor,
    x1_scale: torch.Tensor,
    x2_scale: torch.Tensor,
    bias: torch.Tensor,
    variant: str = "GB_STATIC_PERGROUP",
    transpose_x1: bool = False,
    transpose_x2: bool = True,
    group_size=(1, 128, 128),
) -> torch.Tensor:
    """Torch golden for quant_batch_matmul_v4, aligned to aclnnQuantMatmulV5 (G-B pergroup 静态场景).

    消费已量化 INT8 x1/x2 + 预存 FLOAT32 scale, 沿 K 按 128 分组 block matmul 累加反量化 + bias, cast bf16.
    规格详见模块顶部「规格对齐说明」。

    x1: [M,K] int8 (已量化)            x2: [N,K] int8 (已量化, transposeX2=true)
    x1_scale: [M,⌈K/128⌉] float32      x2_scale: [⌈N/128⌉,⌈K/128⌉] float32
    bias: [N] float32                  out: [M,N] bfloat16
    """
    if variant != "GB_STATIC_PERGROUP":
        raise ValueError("This benchmark fixes variant=GB_STATIC_PERGROUP (static G-B)")
    a = x1.t() if transpose_x1 else x1
    b = x2.t() if transpose_x2 else x2
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    m, k = a.shape
    k2, n = b.shape
    if k != k2 or bias.shape != (n,):
        raise ValueError("shape mismatch")
    # 主线 G-B 硬约束(文档 L662-663): K%512==0(4*128) 且 N%256==0
    if k % 512 != 0 or n % 256 != 0:
        raise ValueError(f"G-B requires K%512==0 and N%256==0 (docs L662-663), got K={k} N={n}")
    gs_k = int(group_size[2])  # 128
    nblk_k = (k + gs_k - 1) // gs_k
    if x1_scale.shape != (m, nblk_k):
        raise ValueError(f"x1_scale expects [{m},{nblk_k}], got {list(x1_scale.shape)}")
    nblk_n = (n + 127) // 128
    if x2_scale.shape != (nblk_n, nblk_k):
        raise ValueError(f"x2_scale expects [{nblk_n},{nblk_k}] (文档: N-group 在前), got {list(x2_scale.shape)}")
    # x2_scale 文档 shape (nblk_n, nblk_k) -> transpose (对齐主线 golden.py: transpose_x2=true 时 scale 也转置)
    # -> (nblk_k, nblk_n) -> repeat N 维 -> (nblk_k, N)
    x2s = x2_scale.to(torch.float32).t().repeat_interleave(128, dim=1)[:, :n]  # [nblk_k, N]
    out = torch.zeros(m, n, dtype=torch.float32, device=x1.device)
    for j in range(nblk_k):
        s = j * gs_k
        e = min((j + 1) * gs_k, k)
        a_blk = a[:, s:e]   # [M, gs_k]
        b_blk = b[s:e, :]   # [gs_k, N]
        partial = a_blk @ b_blk   # [M, N], INT8×INT8 int32 累加(fp32 表达)
        out = out + partial * x1_scale[:, j].reshape(m, 1) * x2s[j, :].reshape(1, n)
    out = out + bias.to(torch.float32).reshape(1, n)
    return out.to(torch.bfloat16)
```
