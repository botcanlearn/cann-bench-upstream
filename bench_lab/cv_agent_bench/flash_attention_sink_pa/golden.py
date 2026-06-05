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

import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        context_lens: torch.Tensor,
        sink_k: torch.Tensor,
        sink_v: torch.Tensor,
    ):
        """
        Args:
            q:            [b, h, 1, d]
            k_cache:      [num_blocks, h, block_size, d]
            v_cache:      [num_blocks, h, block_size, d]
            block_table:  [b, max_blocks_per_seq]  physical block indices
            context_lens: [b]  actual kv sequence length per batch item
            sink_k:       [b, h, sink_size, d]
            sink_v:       [b, h, sink_size, d]
        Returns:
            o:            [b, h, 1, d]
        """
        b, h, _, d = q.shape
        block_size = k_cache.shape[2]
        scale = (1.0 / d) ** 0.5

        outputs = []
        for i in range(b):
            ctx_len = int(context_lens[i].item())
            num_blocks = (ctx_len + block_size - 1) // block_size
            blocks = block_table[i, :num_blocks]  # [num_blocks]

            # gather paged k/v: [h, ctx_len, d]
            k_pages = k_cache[blocks]  # [num_blocks, h, block_size, d]
            v_pages = v_cache[blocks]

            # reshape to [h, num_blocks*block_size, d] then slice to ctx_len
            k_flat = k_pages.permute(1, 0, 2, 3).reshape(h, num_blocks * block_size, d)
            v_flat = v_pages.permute(1, 0, 2, 3).reshape(h, num_blocks * block_size, d)
            k_local = k_flat[:, :ctx_len, :]  # [h, ctx_len, d]
            v_local = v_flat[:, :ctx_len, :]

            qi = q[i]        # [h, 1, d]
            sk = sink_k[i]   # [h, sink_size, d]
            sv = sink_v[i]

            sink_score = torch.einsum("hsd,hkd->hsk", qi, sk) * scale   # [h, 1, sink_size]
            local_score = torch.einsum("hsd,hkd->hsk", qi, k_local) * scale  # [h, 1, ctx_len]

            acc = torch.cat([sink_score, local_score], dim=-1)  # [h, 1, sink_size+ctx_len]
            acc = acc.softmax(dim=-1)

            sink_size = sk.shape[1]
            oi = (torch.einsum("hsk,hkd->hsd", acc[..., :sink_size], sv) +
                  torch.einsum("hsk,hkd->hsd", acc[..., sink_size:], v_local))  # [h, 1, d]
            outputs.append(oi)

        return torch.stack(outputs, dim=0)


def flash_attention_sink_pa(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, block_table: torch.Tensor, context_lens: torch.Tensor, sink_k: torch.Tensor, sink_v: torch.Tensor) -> torch.Tensor:
    """Torch golden implementation aligned to the tile2asc reference Model.forward."""
    model = Model()
    return model(q, k_cache, v_cache, block_table, context_lens, sink_k, sink_v)


def get_input(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
              block_table: torch.Tensor, context_lens: torch.Tensor,
              sink_k: torch.Tensor, sink_v: torch.Tensor,
              seq_len_kv: int = None, **kwargs):
    batch = q.shape[0]
    block_size = k_cache.shape[2]
    table_width = block_table.shape[1]
    requested_len = int(seq_len_kv) if seq_len_kv is not None else table_width * block_size
    legal_len = max(1, min(requested_len, table_width * block_size))
    blocks_needed = max(1, (legal_len + block_size - 1) // block_size)
    total_blocks = k_cache.shape[0]

    rows = []
    for b_idx in range(batch):
        start = b_idx * blocks_needed
        row = (torch.arange(table_width, device=k_cache.device, dtype=torch.int64) + start) % total_blocks
        rows.append(row)
    legal_table = torch.stack(rows, dim=0).contiguous()
    legal_lens = torch.full((batch,), legal_len, dtype=context_lens.dtype, device=context_lens.device)
    return q, k_cache, v_cache, legal_table.to(dtype=block_table.dtype, device=block_table.device), legal_lens, sink_k, sink_v
