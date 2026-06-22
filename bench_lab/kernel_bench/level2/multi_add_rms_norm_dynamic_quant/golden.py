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
from typing import Optional, List


def multi_add_rms_norm_dynamic_quant(
    x1: List[torch.Tensor],
    x2: torch.Tensor,
    gamma: torch.Tensor,
    smooth_scale1: Optional[torch.Tensor] = None,
    smooth_scale2: Optional[torch.Tensor] = None,
    epsilon: float = 1e-6
):
    """
    MultiAddRmsNormDynamicQuant golden implementation.

    Fuses Add + RmsNorm + DynamicQuant:
      1. x = x1 + x2
      2. y = RmsNorm(x, gamma, epsilon)
      3. y1, scale1 = DynamicQuant(y, smooth1)  (smooth1 optional)
      4. y2, scale2 = DynamicQuant(y, smooth2)  (smooth2 optional)

    Returns:
      (y1, y2, x, y, scale1, scale2)
      - y1: int8, quantized output path 1
      - y2: int8, quantized output path 2 (zeros if smooth2 absent)
      - x:  original dtype, x1 + x2
      - y:  original dtype, RmsNorm result
      - scale1: float32, per-row quant scale path 1
      - scale2: float32, per-row quant scale path 2 (zeros if smooth2 absent)
    """
    ori_dtype = x2.dtype

    # Convert to float32 for computation
    x1_fp32 = [t.float() for t in x1]
    x2_fp32 = x2.float()
    gamma_fp32 = gamma.float()

    # Step 1: Add all x1 tensors with x2
    x_sum = x2_fp32.clone()
    for t in x1_fp32:
        x_sum = x_sum + t

    # Step 2: RmsNorm
    rstd = torch.rsqrt(x_sum.pow(2).mean(dim=-1, keepdim=True) + epsilon)
    y_fp32 = x_sum * rstd * gamma_fp32

    # Step 3: DynamicQuant path 1
    if smooth_scale1 is not None:
        input1 = y_fp32 * smooth_scale1.float()
    else:
        input1 = y_fp32

    x_max1 = torch.max(torch.abs(input1), dim=-1, keepdim=True)[0]
    x_max1 = torch.clamp(x_max1, min=1e-8)#防止除零
    gs_rev1 = 127.0 / x_max1
    scale1 = x_max1 / 127.0
    y1 = torch.round(input1 * gs_rev1).to(torch.int8)

    # Step 4: DynamicQuant path 2
    if smooth_scale2 is not None:
        input2 = y_fp32 * smooth_scale2.float()
        x_max2 = torch.max(torch.abs(input2), dim=-1, keepdim=True)[0]
        gs_rev2 = 127.0 / x_max2
        scale2 = x_max2 / 127.0
        y2 = torch.round(input2 * gs_rev2).to(torch.int8)
    else:
        y2 = torch.zeros_like(y1)
        scale2 = torch.zeros_like(scale1)

    # Convert outputs to original dtype
    x_out = x_sum.to(ori_dtype)
    y_out = y_fp32.to(ori_dtype)

    # Remove keepdim from scale
    scale1_out = scale1.squeeze(-1).float()
    scale2_out = scale2.squeeze(-1).float()

    return y1, y2, x_out, y_out, scale1_out, scale2_out
