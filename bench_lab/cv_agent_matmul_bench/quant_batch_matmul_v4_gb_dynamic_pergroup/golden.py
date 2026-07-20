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


def quant_batch_matmul_v4(
    x1: torch.Tensor,
    x2: torch.Tensor,
    bias: torch.Tensor,
    variant: str = "GB_DYNAMIC_PERGROUP",
    transpose_x1: bool = False,
    transpose_x2: bool = False,
    group_size=(1, 128, 128),
) -> torch.Tensor:
    """Torch golden for QuantBatchMatmulV4 G-B dynamic per-group path."""
    if variant != "GB_DYNAMIC_PERGROUP":
        raise ValueError("This benchmark fixes variant=GB_DYNAMIC_PERGROUP")
    a = x1.t() if transpose_x1 else x1
    b = x2.t() if transpose_x2 else x2
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    m, k = a.shape
    k2, n = b.shape
    if k != k2 or bias.shape != (n,):
        raise ValueError("shape mismatch")
    gs_k = int(group_size[2])
    eps = torch.finfo(torch.float32).tiny
    qmax = 127.0
    out = torch.zeros(m, n, dtype=torch.float32, device=x1.device)
    for start in range(0, k, gs_k):
        end = min(start + gs_k, k)
        a_blk = a[:, start:end]
        b_blk = b[start:end, :]
        s1 = a_blk.abs().amax(dim=1, keepdim=True).clamp_min(eps) / qmax
        s2 = b_blk.abs().amax(dim=0, keepdim=True).clamp_min(eps) / qmax
        a_q = torch.round(a_blk / s1).clamp(-qmax, qmax)
        b_q = torch.round(b_blk / s2).clamp(-qmax, qmax)
        out = out + (a_q @ b_q) * s1 * s2
    return out + bias.to(torch.float32).reshape(1, n)
