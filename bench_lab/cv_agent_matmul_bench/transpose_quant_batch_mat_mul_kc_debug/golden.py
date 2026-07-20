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
    y = torch.matmul(a, b)
    y = y * x1Scale.to(torch.float32).reshape(batch, m, 1) * x2Scale.to(torch.float32).reshape(batch, 1, n)
    y = y + bias.to(torch.float32).reshape(batch, 1, n)
    return y.permute(*permY)
