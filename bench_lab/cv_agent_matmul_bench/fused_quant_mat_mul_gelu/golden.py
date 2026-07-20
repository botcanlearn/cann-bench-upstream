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


def _gelu_erf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.erf(x / 1.4142135623730951))


def _gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


def fused_quant_mat_mul(
    x1: torch.Tensor,
    x2: torch.Tensor,
    x1Scale: torch.Tensor,
    x2Scale: torch.Tensor,
    bias: torch.Tensor,
    fused_op_type: str = "gelu_erf",
    transpose_x1: bool = False,
    transpose_x2: bool = False,
    y_dtype: str = "float32",
) -> torch.Tensor:
    """Torch golden for fused_quant_mat_mul GELU path."""
    if transpose_x1:
        x1 = x1.transpose(-2, -1)
    if transpose_x2:
        x2 = x2.transpose(-2, -1)
    m, k = x1.shape
    k2, n = x2.shape
    if k != k2:
        raise ValueError("K mismatch")
    qbmm = (x1.to(torch.float32) @ x2.to(torch.float32))
    qbmm = qbmm * x1Scale.to(torch.float32).reshape(m, 1) * x2Scale.to(torch.float32).reshape(1, n)
    qbmm = qbmm + bias.to(torch.float32).reshape(1, n)
    if fused_op_type == "gelu_erf":
        return _gelu_erf(qbmm)
    if fused_op_type == "gelu_tanh":
        return _gelu_tanh(qbmm)
    raise ValueError("fused_op_type must be gelu_erf or gelu_tanh")
