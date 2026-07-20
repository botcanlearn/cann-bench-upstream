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
    """
    AFT-Full (Attention Free Transformer - Full variant): computes attention-free
    token mixing using element-wise operations with learned position biases.
    Uses sigmoid gating on queries and exponential weighting on keys with position biases.
    """

    def __init__(self, d_model, n=49):
        """
        :param d_model: Dimensionality of the model
        :param n: Sequence length (number of tokens)
        """
        super(Model, self).__init__()
        self.position_biases = nn.Parameter(torch.ones((n, n)))
        self.d_model = d_model
        self.n = n
        self.sigmoid = nn.Sigmoid()

    def forward(self, q, k, v):
        """
        :param q: Query tensor (bs, n, dim), already linear-mapped
        :param k: Key tensor (bs, n, dim), already linear-mapped
        :param v: Value tensor (bs, n, dim), already linear-mapped
        :return: Output tensor (bs, n, dim)
        """
        bs, n, dim = q.shape

        k = k.view(1, bs, n, dim)  # 1, bs, n, dim
        v = v.view(1, bs, n, dim)  # 1, bs, n, dim

        numerator = torch.sum(torch.exp(k + self.position_biases.view(n, 1, -1, 1)) * v, dim=2)  # n, bs, dim
        denominator = torch.sum(torch.exp(k + self.position_biases.view(n, 1, -1, 1)), dim=2)  # n, bs, dim

        out = (numerator / denominator)  # n, bs, dim
        out = self.sigmoid(q) * (out.permute(1, 0, 2))  # bs, n, dim

        return out


def aft_full(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Torch golden implementation aligned to the tile2asc reference Model.forward."""
    model = Model(q.shape[-1], q.shape[1])
    return model(q, k, v)
