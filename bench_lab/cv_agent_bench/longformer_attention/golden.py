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

class Model(nn.Module):
    """
    Longformer Attention mechanism.
    Combines local sliding window attention with global attention on selected tokens.
    """

    def __init__(self, d_model, n_heads, window_size):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.global_attention_indices = [0, 511]
        self.d_k = d_model // n_heads

        self.dropout = nn.Dropout(p=0.0)

    def create_longformer_mask(self, seq_len, window_size, global_indices, device):
        mask = torch.zeros(seq_len, seq_len, device=device)

        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask[i, start:end] = 1

            if i in global_indices:
                mask[i, :] = 1
                mask[:, i] = 1

        return mask

    def forward(self, Q, K, V):
        """
        Args:
            Q: [batch_size, n_heads, seq_len, d_k]
            K: [batch_size, n_heads, seq_len, d_k]
            V: [batch_size, n_heads, seq_len, d_k]

        Returns:
            output: [batch_size, n_heads, seq_len, d_k]
        """
        batch_size, n_heads, seq_len, d_k = Q.size()

        mask = self.create_longformer_mask(
            seq_len, self.window_size, self.global_attention_indices, Q.device
        )
        mask = mask.unsqueeze(0).unsqueeze(0)

        scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V.float())

        return output


def longformer_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, window_size: int = 32) -> torch.Tensor:
    """Torch golden implementation aligned to the tile2asc reference Model.forward."""
    model = Model(Q.shape[1] * Q.shape[-1], Q.shape[1], window_size)
    return model(Q, K, V)
