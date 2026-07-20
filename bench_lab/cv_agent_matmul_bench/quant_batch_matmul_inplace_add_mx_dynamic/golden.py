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


def quant_batch_matmul_inplace_add(
    x1: torch.Tensor,
    x2: torch.Tensor,
    yRef: torch.Tensor,
    variant: str = "MX_DYNAMIC",
    transposeX1: bool = True,
    transposeX2: bool = False,
    groupSize: int = 32,
) -> torch.Tensor:
    """Torch golden for quant_batch_matmul_inplace_add MX dynamic path."""
    if variant != "MX_DYNAMIC":
        raise ValueError("This benchmark fixes variant=MX_DYNAMIC")
    a = x1.t() if transposeX1 else x1
    b = x2.t() if transposeX2 else x2
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    m, k = a.shape
    k2, n = b.shape
    if k != k2 or yRef.shape != (m, n):
        raise ValueError("shape mismatch")
    eps = torch.finfo(torch.float32).tiny
    qmax = 127.0
    out = yRef.to(torch.float32).clone()
    for start in range(0, k, int(groupSize)):
        end = min(start + int(groupSize), k)
        a_blk = a[:, start:end]
        b_blk = b[start:end, :]
        s1 = a_blk.abs().amax(dim=1, keepdim=True).clamp_min(eps) / qmax
        s2 = b_blk.abs().amax(dim=0, keepdim=True).clamp_min(eps) / qmax
        a_dq = torch.round(a_blk / s1).clamp(-qmax, qmax) * s1
        b_dq = torch.round(b_blk / s2).clamp(-qmax, qmax) * s2
        out = out + a_dq @ b_dq
    return out
