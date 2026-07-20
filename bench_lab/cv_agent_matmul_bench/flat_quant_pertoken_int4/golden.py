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


def flat_quant(
    x: torch.Tensor,
    kroneckerP1: torch.Tensor,
    kroneckerP2: torch.Tensor,
    clipRatio: float = 1.0,
    quant_mode: str = "pertoken",
    out_dtype: str = "int4_logical",
):
    """Torch golden for flat_quant per-token logical INT4 path."""
    if quant_mode != "pertoken" or out_dtype != "int4_logical":
        raise ValueError("This benchmark fixes pertoken logical INT4 output")
    if x.dim() != 3:
        raise ValueError("x must be [K,M,N]")
    k, m, n = x.shape
    if kroneckerP1.shape != (m, m) or kroneckerP2.shape != (n, n):
        raise ValueError("kronecker matrices must be [M,M] and [N,N]")
    tmp = torch.einsum('ab,kbn->kan', kroneckerP1.to(torch.float32), x.to(torch.float32))
    transformed = torch.einsum('kmn,nc->kmc', tmp, kroneckerP2.to(torch.float32))
    max_abs = transformed.abs().amax(dim=(1, 2), keepdim=True)
    denom = 7.0 / float(clipRatio)
    scale = max_abs / denom
    normalized = torch.where(scale > 0, transformed / scale, torch.zeros_like(transformed))
    out = torch.round(normalized).clamp(-7, 7).to(torch.int8)
    return out, scale.reshape(k).to(torch.float32)
