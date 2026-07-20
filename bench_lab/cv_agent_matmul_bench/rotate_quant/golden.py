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

    x_fp32 = x.to(torch.float32)
    rot_fp32 = rotation.to(torch.float32)
    y_rot = torch.matmul(x_fp32.reshape(m, n // k, k), rot_fp32).reshape(m, n)

    c_max = 127.0
    max_abs = torch.abs(y_rot).amax(dim=-1, keepdim=True)
    scale = max_abs / c_max
    normalized = torch.where(scale > 0, y_rot / scale, torch.zeros_like(y_rot))
    y = torch.round(normalized).clamp(-c_max, c_max).to(torch.int8)
    return y, scale.reshape(m).to(torch.float32)
