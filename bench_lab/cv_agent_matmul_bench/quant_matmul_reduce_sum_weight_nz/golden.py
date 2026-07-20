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


def quant_matmul_reduce_sum(
    x1: torch.Tensor,
    x2: torch.Tensor,
    x1Scale: torch.Tensor,
    x2Scale: torch.Tensor,
    dims=(0,),
    keep_dims: bool = False,
    x2_format: str = "NZ",
) -> torch.Tensor:
    """Torch golden for quant_matmul_reduce_sum with x2 in NZ layout."""
    if tuple(dims) != (0,):
        raise ValueError("This benchmark fixes dims=[0]")
    if keep_dims:
        raise ValueError("This benchmark fixes keep_dims=False")
    if str(x2_format).upper() != "NZ":
        raise ValueError("This benchmark fixes x2_format=NZ")
    if x1.dim() != 3:
        raise ValueError(f"x1 expects 3D [B,M,K], got {list(x1.shape)}")
    if x2.dim() != 5:
        raise ValueError(f"x2 expects 5D NZ [B,N1,K1,16,32], got {list(x2.shape)}")

    b, m, k = x1.shape
    n = x2Scale.numel()
    if x1Scale.shape != (b, m):
        raise ValueError(f"x1Scale expects shape [{b}, {m}], got {list(x1Scale.shape)}")

    x2_nd = _nz_weight_to_nd(x2, b, k, n)
    mm = torch.matmul(x1.to(torch.float32), x2_nd.to(torch.float32))
    mm = mm * x1Scale.to(torch.float32).reshape(b, m, 1)
    mm = mm * x2Scale.to(torch.float32).reshape(1, 1, n)
    out = mm.sum(dim=0)
    return out.to(torch.bfloat16)


def _nz_weight_to_nd(x2: torch.Tensor, batch: int, k: int, n: int) -> torch.Tensor:
    b, n1, k1, k0, n0 = x2.shape
    if b != batch:
        raise ValueError(f"x2 batch ({b}) must match x1 batch ({batch})")
    if k0 != 16 or n0 != 32:
        raise ValueError(f"NZ x2 expects k0=16,n0=32, got k0={k0}, n0={n0}")
    if k1 != (k + 15) // 16:
        raise ValueError(f"x2 K1 ({k1}) must equal ceil(K/16) for K={k}")
    if n1 != (n + 31) // 32:
        raise ValueError(f"x2 N1 ({n1}) must equal ceil(N/32) for N={n}")
    nd = x2.permute(0, 2, 3, 1, 4).contiguous().reshape(b, k1 * k0, n1 * n0)
    return nd[:, :k, :n]
