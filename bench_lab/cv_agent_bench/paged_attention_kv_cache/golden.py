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

import math

class Model(nn.Module):
    """
    Paged KV Cache Attention for LLM inference decoding.

    Uses paged (block-based) memory management for the KV cache, supporting
    Grouped Query Attention (GQA). Each batch element has a page table mapping
    logical block indices to physical blocks in the cache.

    Input Q:          (batch, seqlen_q, nheads_q, headdim), bfloat16
    Input k_cache:    (num_blocks, page_block_size, nheads_kv, headdim), bfloat16
    Input v_cache:    (num_blocks, page_block_size, nheads_kv, headdim), bfloat16
    Input cache_seqlens: (batch,), int32
    Input page_table: (batch, max_num_blocks_per_seq), int32
    Output:           (batch, seqlen_q, nheads_q, headdim), bfloat16
    """

    def __init__(self, nheads_q, nheads_kv, headdim, page_block_size, causal=True):
        super().__init__()
        self.nheads_q = nheads_q
        self.nheads_kv = nheads_kv
        self.headdim = headdim
        self.page_block_size = page_block_size
        self.causal = causal
        self.scale = 1.0 / math.sqrt(headdim)
        assert nheads_q % nheads_kv == 0
        self.n_groups = nheads_q // nheads_kv

    def _reconstruct_kv(self, cache, cache_seqlen, page_table_row):
        """Reconstruct contiguous KV from paged cache for a single batch element."""
        seq_len = cache_seqlen.item()
        num_blocks_needed = (seq_len + self.page_block_size - 1) // self.page_block_size
        kv_parts = []
        for block_idx in range(num_blocks_needed):
            physical_block = page_table_row[block_idx].item()
            if block_idx == num_blocks_needed - 1:
                remaining = seq_len - block_idx * self.page_block_size
                kv_parts.append(cache[physical_block, :remaining])
            else:
                kv_parts.append(cache[physical_block])
        return torch.cat(kv_parts, dim=0)  # (seq_len, nheads_kv, headdim)

    def forward(self, q, k_cache, v_cache, cache_seqlens, page_table):
        batch = q.shape[0]
        seqlen_q = q.shape[1]

        outputs = []
        for b in range(batch):
            seq_len = cache_seqlens[b].item()

            # Reconstruct K, V from paged cache
            k = self._reconstruct_kv(k_cache, cache_seqlens[b], page_table[b])
            v = self._reconstruct_kv(v_cache, cache_seqlens[b], page_table[b])

            # Reshape for attention: (1, nheads_kv, seq_len, headdim)
            k = k.unsqueeze(0).transpose(1, 2).float()
            v = v.unsqueeze(0).transpose(1, 2).float()

            # GQA: expand KV heads to match Q heads
            if self.n_groups > 1:
                k = k.repeat_interleave(self.n_groups, dim=1)
                v = v.repeat_interleave(self.n_groups, dim=1)

            q_b = q[b:b+1].transpose(1, 2).float()  # (1, nheads_q, seqlen_q, headdim)

            # Attention scores
            scores = torch.matmul(q_b, k.transpose(-2, -1)) * self.scale

            # Causal mask
            if self.causal:
                row_idx = torch.arange(seqlen_q, device=q.device).unsqueeze(1)
                col_idx = torch.arange(seq_len, device=q.device).unsqueeze(0)
                causal_mask = col_idx > (row_idx + seq_len - seqlen_q)
                scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

            attn_weights = torch.softmax(scores, dim=-1)
            out_b = torch.matmul(attn_weights, v)
            out_b = out_b.transpose(1, 2).to(q.dtype)  # (1, seqlen_q, nheads_q, headdim)
            outputs.append(out_b)

        return torch.cat(outputs, dim=0)


def paged_attention_kv_cache(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, cache_seqlens: torch.Tensor, page_table: torch.Tensor, causal: bool = True) -> torch.Tensor:
    """Torch golden implementation aligned to the tile2asc reference Model.forward."""
    model = Model(q.shape[2], k_cache.shape[2], q.shape[-1], k_cache.shape[1], causal)
    return model(q, k_cache, v_cache, cache_seqlens, page_table)


def get_input(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
              cache_seqlens: torch.Tensor, page_table: torch.Tensor,
              cache_seqlen: int = None, **kwargs):
    batch, seqlen_q = q.shape[:2]
    page_block_size = k_cache.shape[1]
    table_width = page_table.shape[1]
    requested_len = int(cache_seqlen) if cache_seqlen is not None else int(cache_seqlens.max().item())
    legal_len = max(1, min(max(requested_len, seqlen_q), table_width * page_block_size))
    blocks_needed = max(1, (legal_len + page_block_size - 1) // page_block_size)
    num_blocks = k_cache.shape[0]

    rows = []
    for b_idx in range(batch):
        start = b_idx * blocks_needed
        row = (torch.arange(table_width, device=k_cache.device, dtype=torch.int64) + start) % num_blocks
        rows.append(row)
    legal_table = torch.stack(rows, dim=0).contiguous()
    legal_lens = torch.full((batch,), legal_len, dtype=cache_seqlens.dtype, device=cache_seqlens.device)
    return q, k_cache, v_cache, legal_lens, legal_table.to(dtype=page_table.dtype, device=page_table.device)
