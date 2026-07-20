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

import math

def create_bigbird_mask(seq_len, window_size, num_random_blocks, device):
    """
    Create a BigBird hybrid mask.
    Includes: local window + global tokens (first/last) + random attention.
    """
    mask = torch.zeros(seq_len, seq_len, device=device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(2026)

    # 1. Local window attention
    for i in range(seq_len):
        start = max(0, i - window_size // 2)
        end = min(seq_len, i + window_size // 2 + 1)
        mask[i, start:end] = 1

    # 2. Global attention (first and last tokens)
    mask[0, :] = 1   # First token attends to all
    mask[-1, :] = 1  # Last token attends to all
    mask[:, 0] = 1   # All attend to first token
    mask[:, -1] = 1  # All attend to last token

    # 3. Random attention
    for i in range(1, seq_len - 1):  # Exclude first and last
        random_indices = torch.randperm(seq_len, generator=generator)[:num_random_blocks].to(device)
        mask[i, random_indices] = 1

    return mask

class Model(nn.Module):
    """
    BigBird Attention mechanism.
    Combines random attention, local sliding window attention, and global attention
    on the first and last tokens.
    """

    def __init__(self, d_model, n_heads, window_size, num_random_blocks):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.num_random_blocks = num_random_blocks
        self.d_k = d_model // n_heads

        self.dropout = nn.Dropout(p=0.0)

    def forward(self, q, k, v, mask):
        """
        Forward pass.

        Args:
            q: query tensor [batch_size, n_heads, seq_len, d_k]
            k: key tensor [batch_size, n_heads, seq_len, d_k]
            v: value tensor [batch_size, n_heads, seq_len, d_k]

        Returns:
            output: attention output [batch_size, n_heads, seq_len, d_k]
        """
        batch_size, _, seq_len, _ = q.size()

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)

        return output


def bigbird_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, window_size: int = 32, num_random_blocks: int = 3) -> torch.Tensor:
    """Torch golden implementation aligned to the tile2asc reference Model.forward."""
    model = Model(q.shape[1] * q.shape[-1], q.shape[1], window_size, num_random_blocks)
    return model(q, k, v, mask)


def get_input(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor,
              window_size: int = 32, num_random_blocks: int = 3, **kwargs):
    seq_len = q.shape[-2]
    legal_mask = create_bigbird_mask(seq_len, window_size, num_random_blocks, q.device)
    return q, k, v, legal_mask.to(dtype=mask.dtype, device=mask.device)
