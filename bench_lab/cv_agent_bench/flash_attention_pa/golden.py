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

def _gather_paged(kv_cache, block_table):
    """通过 block_table 将分页 KV Cache 映射为连续张量 [B, H, S, D]。"""
    gathered = kv_cache[block_table]  # [B, num_logical_blocks, block_size, H, D]
    batch, num_blocks = block_table.shape
    block_size, heads, head_dim = kv_cache.shape[1], kv_cache.shape[2], kv_cache.shape[3]
    seq_len = num_blocks * block_size
    return gathered.reshape(batch, seq_len, heads, head_dim).permute(0, 2, 1, 3)

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, q, k_cache, v_cache, block_table):
        k = _gather_paged(k_cache, block_table)
        v = _gather_paged(v_cache, block_table)
        acc = torch.einsum("bhsd,bhkd->bhsk", q, k) * (1.0 / q.shape[-1]) ** 0.5
        acc = acc.softmax(dim=-1)
        o = torch.einsum("bhsk,bhkd->bhsd", acc, v)
        return o.to(torch.float16)


def flash_attention_pa(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, block_table: torch.Tensor) -> torch.Tensor:
    """Torch golden implementation aligned to the tile2asc reference Model.forward."""
    model = Model()
    return model(q, k_cache, v_cache, block_table)


def get_input(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
              block_table: torch.Tensor, **kwargs):
    batch, _, seq_len, _ = q.shape
    block_size = k_cache.shape[1]
    logical_blocks = max(1, (seq_len + block_size - 1) // block_size)
    total_blocks = k_cache.shape[0]
    table_width = max(block_table.shape[1], logical_blocks)
    rows = []
    for b_idx in range(batch):
        start = b_idx * logical_blocks
        row = (torch.arange(table_width, device=k_cache.device, dtype=torch.int64) + start) % total_blocks
        rows.append(row)
    legal_table = torch.stack(rows, dim=0)[:, :logical_blocks].contiguous()
    return q, k_cache, v_cache, legal_table.to(dtype=block_table.dtype, device=block_table.device)
