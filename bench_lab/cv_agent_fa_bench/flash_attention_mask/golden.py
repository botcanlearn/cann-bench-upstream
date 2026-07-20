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

import torch.nn as nn

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor):
        q_fp32 = q.to(torch.float32)
        k_fp32 = k.to(torch.float32)
        v_fp32 = v.to(torch.float32)
        acc = torch.einsum("bhsd,bhkd->bhsk", q_fp32, k_fp32) * (1.0 / q.shape[-1]) ** 0.5
        mask_bool = mask.to(torch.bool)
        acc = acc.masked_fill(~mask_bool, -torch.inf)
        acc = acc.softmax(dim=-1)
        o = torch.einsum("bhsk,bhkd->bhsd", acc, v_fp32)
        return o.to(torch.float16)


def flash_attention_mask(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Torch golden implementation aligned to the tile2asc reference Model.forward."""
    model = Model()
    return model(q, k, v, mask)


def get_input(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor,
              mask_type: str = "full", **kwargs):
    batch, heads, seq_len_q, _ = q.shape
    seq_len_kv = k.shape[-2]
    row = torch.arange(seq_len_q, device=q.device).unsqueeze(1)
    col = torch.arange(seq_len_kv, device=q.device).unsqueeze(0)

    if mask_type == "causal":
        base = col <= row + max(seq_len_kv - seq_len_q, 0)
    elif mask_type == "band":
        center = row + max(seq_len_kv - seq_len_q, 0)
        radius = max(1, min(seq_len_q, seq_len_kv) // 8)
        base = (col >= center - radius) & (col <= center + radius)
    elif mask_type == "block":
        block = max(1, min(seq_len_q, seq_len_kv) // 4)
        base = (row // block) == (col // block)
    else:
        base = torch.ones((seq_len_q, seq_len_kv), dtype=torch.bool, device=q.device)

    # Ensure every query row has at least one visible KV position.
    empty_rows = ~base.any(dim=-1)
    if empty_rows.any():
        base[empty_rows, 0] = True

    legal_mask = base.unsqueeze(0).unsqueeze(0).expand(batch, heads, -1, -1).contiguous()
    return q, k, v, legal_mask.to(dtype=mask.dtype, device=mask.device)
