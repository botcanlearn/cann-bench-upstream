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
from typing import Tuple


def rmsnorm(x, gamma, eps=1e-6):
    """RMS Normalization (keeps fp32, no intermediate dtype drop)."""
    if x.dtype in (torch.float16, torch.bfloat16):
        x = x.float()
        gamma = gamma.float()
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return x / rms * gamma


def fused_rmsnorm_pos_qkv_qknorm(
    x: torch.Tensor, gamma: torch.Tensor, pos_emb: torch.Tensor,
    Wqkv: torch.Tensor, gamma_q: torch.Tensor, gamma_k: torch.Tensor,
    num_heads: int = 4, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Golden reference: RMSNorm + pos + QKV projection + QK norm.

    Args:
        x: input hidden states [B, S, D]
        gamma: RMSNorm weight [D]
        pos_emb: positional embedding [1, S, D]
        Wqkv: QKV projection weight [D, 3*H]
        gamma_q: Q normalization weight [H]
        gamma_k: K normalization weight [H]
        num_heads: number of attention heads
        eps: epsilon for RMSNorm

    Returns:
        q: normalized Q [B, S, H]
        k: normalized K [B, S, H]
        v: V projection [B, S, H]
    """
    original_dtype = x.dtype
    B, S, D = x.shape
    H = Wqkv.shape[1] // 3
    assert H % num_heads == 0, f"H={H} must be divisible by num_heads={num_heads}"

    h = rmsnorm(x, gamma, eps) + pos_emb.float()

    qkv = torch.matmul(h, Wqkv.float())
    q, k, v = qkv.split(H, dim=-1)

    q = rmsnorm(q, gamma_q, eps)
    k = rmsnorm(k, gamma_k, eps)

    return q.to(original_dtype), k.to(original_dtype), v.to(original_dtype)
