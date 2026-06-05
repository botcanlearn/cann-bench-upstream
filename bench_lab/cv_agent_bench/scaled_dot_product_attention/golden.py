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
    Scaled Dot-Product Attention.
    Computes Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    """

    def __init__(self, d_k, dropout=0.0):
        """
        Args:
            d_k: Key dimension.
            dropout: Dropout probability (default 0.0 for inference).
        """
        super().__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V):
        """
        Forward pass.

        Args:
            Q: Query tensor [batch_size, n_heads, seq_len, d_k]
            K: Key tensor [batch_size, n_heads, seq_len, d_k]
            V: Value tensor [batch_size, n_heads, seq_len, d_v]

        Returns:
            output: Attention output [batch_size, n_heads, seq_len, d_v]
        """
        batch_size, n_heads, seq_len, d_k = Q.size()

        # Compute attention scores: QK^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)

        # Apply dropout
        attn_weights = self.dropout(attn_weights)

        # Compute output: attention weights x V
        output = torch.matmul(attn_weights, V)

        return output


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, dropout: float = 0.0) -> torch.Tensor:
    """Torch golden implementation aligned to the tile2asc reference Model.forward."""
    model = Model(Q.shape[-1], dropout)
    return model(Q, K, V)
