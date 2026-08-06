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


def rotate_quant(
    x: torch.Tensor,
    rotation: torch.Tensor,
    alpha: float = 0.0,
    y_dtype: str = "int8",
):
    """Torch golden for the selected rotate_quant int8 path."""
    if str(y_dtype).lower() != "int8":
        raise ValueError("This benchmark fixes rotate_quant y_dtype=int8")
    if alpha != 0.0:
        raise ValueError("This benchmark fixes rotate_quant alpha=0.0")
    if x.dim() != 2:
        raise ValueError(f"rotate_quant expects x to be 2D, got shape {list(x.shape)}")
    if rotation.dim() != 2 or rotation.shape[0] != rotation.shape[1]:
        raise ValueError(f"rotation must be square, got shape {list(rotation.shape)}")

    m, n = x.shape
    k = rotation.shape[0]
    if n % k != 0:
        raise ValueError(f"N ({n}) must be divisible by K ({k})")
    if n % 8 != 0:
        raise ValueError(f"N ({n}) must be divisible by 8")
    if n < 128 or n > 16000:
        raise ValueError(f"N ({n}) must be in [128, 16000]")
    if k < 16 or k > 1024:
        raise ValueError(f"K ({k}) must be in [16, 1024]")

    # AIC computes the rotation matmul with the cube unit's internal fp32
    # accumulator, but MatmulImpl's C tensor is typed DTYPE_X (see
    # op_kernel/rotate_quant.cpp: `using cType = MatmulType<..., DTYPE_X>`),
    # so the rotated result is truncated to x.dtype (fp16/bf16) when written
    # to the workspace GM buffer. The AIV quant stage then reads it back and
    # up-casts to fp32 with RoundMode::CAST_NONE (op_kernel/rotate_quant.h
    # `CopyInVector`). Skipping this fp16/bf16 round-trip overstates the
    # precision of the rotated activations before quantization.
    x_fp32 = x.to(torch.float32)
    rot_fp32 = rotation.to(torch.float32)
    y_rot_fp32 = torch.matmul(x_fp32.reshape(m, n // k, k), rot_fp32).reshape(m, n)
    y_rot = y_rot_fp32.to(x.dtype).to(torch.float32)

    # Per-row symmetric dynamic quant, matching op_kernel/rotate_quant.h AIV stage:
    #   scaleTmp      = amax_j |Y[i, j]|                       (ComputeReduceMax)
    #   quantScaleTmp = 127.0 / scaleTmp                        (Div constScale/scaleTmp)
    #   normalized    = Y * quantScaleTmp                       (Mul, broadcast)
    #   y             = round-to-nearest-even(normalized)       (Cast CAST_RINT)
    #   scaleOut      = scaleTmp * (1/127)                      (Mul constInvScale)
    # The reciprocal-multiply form (not Y / (amax/127)) and the 1/127 constant
    # are reproduced so fp32 rounding matches the kernel rather than an
    # algebraically-equal but numerically-different division.
    c_max = 127.0
    inv_c_max = float(1.0) / c_max
    max_abs = torch.abs(y_rot).amax(dim=-1, keepdim=True)
    quant_scale = torch.where(max_abs > 0, c_max / max_abs, torch.zeros_like(max_abs))
    normalized = y_rot * quant_scale
    # CAST_RINT is round-half-to-even; the symmetric scale keeps values within
    # [-127, 127] by construction, so the clamp is a defensive no-op.
    y = torch.round(normalized).clamp(-c_max, c_max).to(torch.int8)
    scale = (max_abs * inv_c_max).reshape(m).to(torch.float32)
    return y, scale
