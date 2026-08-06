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
