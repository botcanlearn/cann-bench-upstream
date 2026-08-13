# QuantBatchMatmulInplaceAdd 算子 API 描述

## 1. 算子简介

`quant_batch_matmul_inplace_add` 对齐主线算子 `aclnnQuantBatchMatmulInplaceAdd`（`ops-nn/matmul/quant_batch_matmul_inplace_add`）的 **MX (mxFP8) 静态量化 + inplace add 场景**。消费**真实 fp8** 的 `x1/x2`(torch.float8_e4m3fn，主线 FLOAT8_E4M3FN)与**预存 E8M0 scale**(uint8 字节，解码 2^(raw-127))，沿 K 按 32 分组 block matmul 乘 scale 反量化，累加到 `yRef`，属于 C->V kernel flow（无在线动态量化）。主线 `transposeX1=true`/`transposeX2=false`、scale 末维=2、仅 950。

## 2. 算子定义

固定 `transposeX1=true`、`transposeX2=false`，`x1` 输入 `[K,M]`（逻辑 `[M,K]`），`x2` 为 `[K,N]`。K 维按 `groupSize=32` 分块：

```text
y = yRef
for each K block j:
    block = x1[Kj,:] @ x2[Kj,:]             # 已量化 fp8 block matmul (不 round)
    y += block * scale1[m,j] * scale2[j,n]  # 预存 E8M0 scale 反量化
```

`scale1`/`scale2` 为预存 E8M0 scale（非在线计算）。

## 3. 接口规范

```python
quant_batch_matmul_inplace_add(x1, x2, x1_scale, x2_scale, yRef, variant="MX_STATIC", transposeX1=True, transposeX2=False, groupSize=32) -> y
```

| 参数 | 输入/输出 | dtype | shape | 说明 |
|------|-----------|-------|-------|------|
| `x1` | 输入 | `FLOAT8_E4M3FN`(真实 torch.float8_e4m3fn) | `[K,M]` | 已量化左矩阵（transposeX1=true） |
| `x2` | 输入 | `FLOAT8_E4M3FN`(真实 torch.float8_e4m3fn) | `[K,N]` | 已量化右矩阵 |
| `x1_scale` | 输入 | `E8M0`(uint8 字节, 解码 2^(raw-127)) | `[⌈K/64⌉,M,2]` | 预存 scale（文档 shape） |
| `x2_scale` | 输入 | `E8M0`(uint8 字节, 解码 2^(raw-127)) | `[⌈K/64⌉,N,2]` | 预存 scale（文档 shape） |
| `yRef` | 输入/输出 | `FLOAT32` | `[M,N]` | inplace add 初始值 |
| `y` | 输出 | `FLOAT32` | `[M,N]` | 累加后的结果 |

## 4. 约束说明

- 固定 `variant=MX_STATIC`、`transposeX1=true`、`transposeX2=false`、`groupSize=32`。
- 输入为**预存静态已量化数据**（FLOAT8 x1/x2 + 预存 E8M0 scale），无在线动态量化。
- 主线 x1/x2=FLOAT8、scale=FLOAT8_E8M0（末维=2）；本 benchmark 用 float32 近似表达（见 golden.py 规格头）。
- K 需为 64 的倍数（E8M0 scale 3D 布局整齐）。

## 5. 精度要求

本算子精度判定遵循 [`benchmark_spec.md` §4.4](../../../docs/spec/benchmark_spec.md)。通过条件与阈值参数定义在同目录 `proto.yaml` 的 `precision` 节点,以下仅说明本算子特定的取舍。

### 5.1 算子特定说明

- **`y` 阈值归属**:规则 `output_dtype`。输出 float32，inplace add 累加。

## 6. 标准 Golden 代码

`golden.py` 按 K 维 32 分块 block matmul，乘预存 E8M0 scale 反量化，inplace 累加到 `yRef`。详见同目录 `golden.py`。

## 7. 额外信息

### 测试资料对应关系

- `docs/aclnnQuantBatchMatmulInplaceAdd.md`：MX 量化公式、`groupSize` 编码、inplace add。
- `op_api/aclnn_quant_batch_matmul_inplace_add.cpp`：E8M0 scale shape（L321-328/L641-643）、groupSize 编码（L207-232）。
- `op_kernel/arch35/qbmmia_mx_*.h`：MX basic API 路径。
- `examples/arch35/test_aclnn_quant_batch_matmul_inplace_add_mxfp8.cpp`：MXFP8 样例。

### 本 benchmark case 设计

`cases.yaml` 当前包含 20 个正向 case，K 为 64 倍数（K∈{64,128,192,256,320,384,512}），覆盖不同 M/N 和 `yRef` 非零累加。

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


# ===== 规格对齐说明(对齐主线 aclnnQuantBatchMatmulInplaceAdd 的 MX mxFP8 静态场景) =====
# 主线算子: aclnnQuantBatchMatmulInplaceAdd (graph op: QuantBatchMatmulInplaceAdd, 仅 ascend950)
#   实现: ops-nn/matmul/quant_batch_matmul_inplace_add
# 对应场景: MX (mxFP8) 量化, groupSizeK=32, inplace add (消费已量化 fp8 + E8M0 scale, 无在线量化)
# 主线公式:
#   gsK = 32
#   y[m,n] = yRef[m,n] + sum_{j=0}^{ceil(K/32)-1} (x1_blk[Kj,m] @ x2_blk[Kj,n]) * (scale1[m,j] * scale2[j,n])
#   (x1/x2 已 fp8 量化, scale 为 E8M0 预存, 无在线 amax/round)
#   出处: docs/aclnnQuantBatchMatmulInplaceAdd.md (MX 公式 L22-28, shape 表 L284, 示例 L403-407);
#         op_api/aclnn_quant_batch_matmul_inplace_add.cpp (L207-232 groupSize编码, L321-328 E8M0 shape);
#         op_kernel/arch35/quant_batch_matmul_inplace_add.cpp (QBMMIA_IS_MX L29-36)
# 主线 dtype/shape 规格:
#   x1        : FLOAT8_E4M3FN/E5M2, 2D (K,M)   (transposeX1=true -> 逻辑 [M,K])
#   x2        : FLOAT8_E4M3FN/E5M2, 2D (K,N)   (transposeX2=false)
#   x1_scale  : FLOAT8_E8M0, 3D (ceil(K/64), M, 2)   [文档 shape; 每 32 K 元素 1 个 E8M0 scale]
#   x2_scale  : FLOAT8_E8M0, 3D (ceil(K/64), N, 2)   [文档 shape]
#   yRef / y  : FLOAT32, 2D (M,N), inplace add
#   groupSize : 32 (编码值 4295032864 = gsK=32 | gsN=1<<16 | gsM=1<<32)
# 关键约束: transposeX1=true && transposeX2=false; E8M0 scale 末维=2; 仅 950PR/950DT.
# 本 golden 与主线的对齐说明:
#   - x1/x2 用【真实 torch.float8_e4m3fn】(本机 torch 2.9 CPU 支持 fp8 native matmul), 不再用 float 近似。
#   - scale 用【uint8 字节(E8M0)】, golden 解码为 2^(raw-127)(严格 2 的幂, 对齐 E8M0 离散性)。
#   - scale 3D 文档 shape (ceil(K/64),*,2) -> golden 内部 permute(0,2,1).reshape(ceil(K/32),*) 提取逻辑 scale, 按 K-group j 索引。
#   - 沿 K 按 32 分组: partial = matmul(x1_blk_fp8, x2_blk_fp8).float(); out += partial*scale1*scale2; inplace 累加 yRef。
# ===========================================================================================


def quant_batch_matmul_inplace_add(
    x1: torch.Tensor,
    x2: torch.Tensor,
    x1_scale: torch.Tensor,
    x2_scale: torch.Tensor,
    yRef: torch.Tensor,
    variant: str = "MX_STATIC",
    transposeX1: bool = True,
    transposeX2: bool = False,
    groupSize: int = 32,
) -> torch.Tensor:
    """Torch golden for quant_batch_matmul_inplace_add, aligned to aclnnQuantBatchMatmulInplaceAdd (MX mxFP8 静态).

    消费真实 fp8 x1/x2 + E8M0(uint8) scale, 沿 K 按 32 分组 block matmul 乘 scale 反量化, inplace 累加 yRef.
    规格详见模块顶部「规格对齐说明」。

    x1: [K,M] float8_e4m3fn           x2: [K,N] float8_e4m3fn
    x1_scale: [⌈K/64⌉,M,2] uint8(E8M0)   x2_scale: [⌈K/64⌉,N,2] uint8(E8M0)
    yRef: [M,N] float32              out: [M,N] float32
    """
    if variant != "MX_STATIC":
        raise ValueError("This benchmark fixes variant=MX_STATIC (static mxFP8)")
    a = x1.t() if transposeX1 else x1   # [M,K] fp8
    b = x2.t() if transposeX2 else x2   # [K,N] fp8
    m, k = a.shape
    k2, n = b.shape
    if k != k2 or yRef.shape != (m, n):
        raise ValueError("shape mismatch")
    if x1_scale.shape[-1] != 2 or x2_scale.shape[-1] != 2:
        raise ValueError(f"E8M0 scale last dim must be 2 (docs L284), got {list(x1_scale.shape)}/{list(x2_scale.shape)}")
    # E8M0 uint8 字节 -> 解码为 2^(raw-127) (严格 2 的幂)
    s1 = torch.pow(2.0, x1_scale.to(torch.float32) - 127.0)
    s2 = torch.pow(2.0, x2_scale.to(torch.float32) - 127.0)
    # 3D (ceil(K/64),*,2) -> permute(0,2,1) -> (ceil(K/64),2,*) -> reshape(ceil(K/32),*)
    ng = (k + groupSize - 1) // groupSize            # ceil(K/32) = num K-groups
    s1 = s1.permute(0, 2, 1).contiguous().reshape(ng, m)
    s2 = s2.permute(0, 2, 1).contiguous().reshape(ng, n)
    out = yRef.to(torch.float32).clone()
    for j in range(ng):
        s = j * groupSize
        e = min((j + 1) * groupSize, k)
        a_blk = a[:, s:e]   # fp8 [M, 32]
        b_blk = b[s:e, :]   # fp8 [32, N]
        partial = a_blk.to(torch.float32) @ b_blk.to(torch.float32)   # fp8->float matmul(累加 fp32, 避免 fp8 native 的 saturation, 对齐主线 L0C fp32 累加)
        out = out + partial * s1[j].reshape(m, 1) * s2[j].reshape(1, n)
    return out
```
