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

import math

import torch

import torch.nn as nn

import torch.nn.functional as F

def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, kv_heads, S, D) -> (B, kv_heads * n_rep, S, D)"""
    if n_rep == 1:
        return x
    B, kv_heads, S, D = x.shape
    return (
        x[:, :, None, :, :]
        .expand(B, kv_heads, n_rep, S, D)
        .contiguous()
        .view(B, kv_heads * n_rep, S, D)
    )

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        q_nope: torch.Tensor,
        q_rope: torch.Tensor,
        k_nope: torch.Tensor,
        k_rope: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        B, H, S_q, d_nope = q_nope.shape
        d_rope = q_rope.shape[-1]
        kv_heads = k_nope.shape[1]

        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        n_rep = H // kv_heads
        k = _repeat_kv(k, n_rep)
        v = _repeat_kv(v, n_rep)

        d_qk = d_nope + d_rope
        acc = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(d_qk))
        acc = F.softmax(acc, dim=-1)
        o = torch.matmul(acc, v)
        return o


def flash_attention_mla(q_nope: torch.Tensor, q_rope: torch.Tensor, k_nope: torch.Tensor, k_rope: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Torch golden implementation aligned to the tile2asc reference Model.forward."""
    model = Model()
    return model(q_nope, q_rope, k_nope, k_rope, v)
