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
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        batch_size, n_heads, seq_len, d_k = q.shape
        assert seq_len % self.block_size == 0
        n_blocks = seq_len // self.block_size

        Q = q.view(batch_size, n_heads, n_blocks, self.block_size, d_k)
        K = k.view(batch_size, n_heads, n_blocks, self.block_size, d_k)
        V = v.view(batch_size, n_heads, n_blocks, self.block_size, d_k)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        output = output.view(batch_size, n_heads, seq_len, d_k)
        return output.to(torch.float16)


def block_sparse_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Torch golden implementation aligned to the tile2asc reference Model.forward."""
    model = Model(block_size)
    return model(q, k, v)
