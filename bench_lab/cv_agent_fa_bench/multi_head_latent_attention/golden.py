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
    Paged single-KV-head latent cache attention.

    The KV cache stores a single-head compressed representation of dimension
    headdim_qk (576). The query has shape (batch, seqlen_q, nheads_q, headdim_qk)
    and the output has a different head dimension headdim_v (512).

    For the attention computation:
      - K scoring uses the full headdim_qk (576) dimensions of the cache
      - V output uses the first headdim_v (512) dimensions of the cache

    Input Q:          (batch, seqlen_q, nheads_q, headdim_qk), bfloat16
    Input kv_cache:   (num_blocks, page_block_size, 1, headdim_qk), bfloat16
    Input block_table:(batch, max_num_blocks_per_seq), int32
    Input cache_seqlens: (batch,), int32
    Output:           (batch, seqlen_q, nheads_q, headdim_v), bfloat16
    """

    def __init__(self, nheads_q, headdim_qk, headdim_v, page_block_size, causal=True):
        super().__init__()
        self.nheads_q = nheads_q
        self.headdim_qk = headdim_qk
        self.headdim_v = headdim_v
        self.page_block_size = page_block_size
        self.causal = causal
        self.scale = 1.0 / math.sqrt(headdim_qk)

    def _reconstruct_from_cache(self, cache, cache_seqlen, block_table_row):
        """Reconstruct contiguous KV from paged cache for one batch element."""
        seq_len = cache_seqlen.item()
        num_blocks_needed = (seq_len + self.page_block_size - 1) // self.page_block_size
        parts = []
        for block_idx in range(num_blocks_needed):
            physical_block = block_table_row[block_idx].item()
            if block_idx == num_blocks_needed - 1:
                remaining = seq_len - block_idx * self.page_block_size
                parts.append(cache[physical_block, :remaining])
            else:
                parts.append(cache[physical_block])
        return torch.cat(parts, dim=0)  # (seq_len, 1, headdim_qk)

    def forward(self, q, kv_cache, block_table, cache_seqlens):
        batch, seqlen_q, nheads_q, headdim_qk = q.shape

        outputs = []
        for b in range(batch):
            seq_len = cache_seqlens[b].item()

            # Reconstruct compressed KV: (seq_len, 1, headdim_qk)
            kv = self._reconstruct_from_cache(
                kv_cache, cache_seqlens[b], block_table[b])

            # K uses full headdim_qk dims for scoring
            # Shape: (1, 1, seq_len, headdim_qk)
            k = kv[:, :, :self.headdim_qk].unsqueeze(0).transpose(1, 2).float()
            # V uses first headdim_v dims for output
            # Shape: (1, 1, seq_len, headdim_v)
            v = kv[:, :, :self.headdim_v].unsqueeze(0).transpose(1, 2).float()

            # Broadcast single KV head to all query heads
            # k: (1, 1, seq_len, headdim_qk) -> broadcasts with (1, nheads_q, seqlen_q, headdim_qk)
            q_b = q[b:b+1].transpose(1, 2).float()  # (1, nheads_q, seqlen_q, headdim_qk)

            # Attention scores: (1, nheads_q, seqlen_q, seq_len)
            scores = torch.matmul(q_b, k.transpose(-2, -1)) * self.scale

            # Causal mask (for dense decoding mode)
            if self.causal:
                row_idx = torch.arange(seqlen_q, device=q.device).unsqueeze(1)
                col_idx = torch.arange(seq_len, device=q.device).unsqueeze(0)
                causal_mask = col_idx > (row_idx + seq_len - seqlen_q)
                scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

            attn_weights = torch.softmax(scores, dim=-1)

            # Output: (1, nheads_q, seqlen_q, headdim_v)
            out_b = torch.matmul(attn_weights, v)
            out_b = out_b.transpose(1, 2).to(q.dtype)  # (1, seqlen_q, nheads_q, headdim_v)
            outputs.append(out_b)

        return torch.cat(outputs, dim=0)


def multi_head_latent_attention(q: torch.Tensor, kv_cache: torch.Tensor, block_table: torch.Tensor, cache_seqlens: torch.Tensor, causal: bool = True) -> torch.Tensor:
    """Torch golden implementation aligned to the tile2asc reference Model.forward."""
    model = Model(q.shape[2], q.shape[-1], 512 if q.shape[-1] > 512 else q.shape[-1], kv_cache.shape[1], causal)
    return model(q, kv_cache, block_table, cache_seqlens)


def get_input(q: torch.Tensor, kv_cache: torch.Tensor, block_table: torch.Tensor,
              cache_seqlens: torch.Tensor, cache_seqlen: int = None, **kwargs):
    batch, seqlen_q = q.shape[:2]
    page_block_size = kv_cache.shape[1]
    table_width = block_table.shape[1]
    requested_len = int(cache_seqlen) if cache_seqlen is not None else int(cache_seqlens.max().item())
    legal_len = max(1, min(max(requested_len, seqlen_q), table_width * page_block_size))
    blocks_needed = max(1, (legal_len + page_block_size - 1) // page_block_size)
    num_blocks = kv_cache.shape[0]

    rows = []
    for b_idx in range(batch):
        start = b_idx * blocks_needed
        row = (torch.arange(table_width, device=kv_cache.device, dtype=torch.int64) + start) % num_blocks
        rows.append(row)
    legal_table = torch.stack(rows, dim=0).contiguous()
    legal_lens = torch.full((batch,), legal_len, dtype=cache_seqlens.dtype, device=cache_seqlens.device)
    return q, kv_cache, legal_table.to(dtype=block_table.dtype, device=block_table.device), legal_lens
