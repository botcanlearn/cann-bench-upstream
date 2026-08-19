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
from typing import Tuple

"""
FlashAttentionBackward 算子 Torch Golden 参考实现

FlashAttention-2 反向传播：由 q/k/v/dO 重算前向 P、O，再按链式法则给出 dQ/dK/dV。
公式（逐 head，S = QK^T·scale，P = softmax(S + causal_mask)）:
    O     = P V
    dV    = P^T dO
    dP    = dO V^T
    Delta = rowsum(dO ⊙ O)
    dS    = P ⊙ (dP − Delta)
    dQ    = dS K · scale
    dK    = dS^T Q · scale
plain golden 内部升 fp32 计算（bench 语义），输出转回输入 dtype；
flash_attention_backward_oracle 跟随输入精度（golden_precision=fp64_cpu 下即为 fp64 真值）。
"""


def _flash_attention_backward_core(query, key, value, dy, scaleValue, is_causal, compute_dtype):
    """核心计算：以 compute_dtype 精度逐 (b, n) 重算前向并求 FA2 反向。"""
    B, N, S, D = query.shape
    q = query.reshape(B * N, S, D).to(compute_dtype)
    k = key.reshape(B * N, S, D).to(compute_dtype)
    v = value.reshape(B * N, S, D).to(compute_dtype)
    do = dy.reshape(B * N, S, D).to(compute_dtype)

    causal_mask = None
    if is_causal:
        # 下三角可见：屏蔽 j > i 的位置（对角线上每行至少 j=i 可见，softmax 恒有效）
        causal_mask = torch.triu(
            torch.ones(S, S, dtype=torch.bool, device=query.device), diagonal=1)

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    # 按 (b, n) 逐头计算，控制 [S, S] 中间矩阵的峰值内存
    for i in range(B * N):
        qi, ki, vi, doi = q[i], k[i], v[i], do[i]
        # === 前向重算 ===
        scores = torch.matmul(qi, ki.transpose(-2, -1)) * scaleValue   # [S, S]
        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask, float('-inf'))
        p = torch.softmax(scores, dim=-1)                              # [S, S]
        o = torch.matmul(p, vi)                                        # [S, D]
        # === FA2 反向公式 ===
        dv[i] = torch.matmul(p.transpose(-2, -1), doi)                 # dV = P^T @ dO
        dp = torch.matmul(doi, vi.transpose(-2, -1))                   # dP = dO @ V^T
        delta = (doi * o).sum(dim=-1, keepdim=True)                    # Delta_i = rowsum(dO ⊙ O)
        ds = p * (dp - delta)                                          # dS = P ⊙ (dP - Delta)
        dq[i] = torch.matmul(ds, ki) * scaleValue                      # dQ = dS @ K * scale
        dk[i] = torch.matmul(ds.transpose(-2, -1), qi) * scaleValue    # dK = dS^T @ Q * scale

    return dq.reshape(B, N, S, D), dk.reshape(B, N, S, D), dv.reshape(B, N, S, D)


def flash_attention_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dy: torch.Tensor,
    scaleValue: float,
    is_causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    FlashAttention 反向传播 golden reference（plain golden = bench：内部 fp32 计算）

    Args:
        query: [B, N, S, D] 前向查询张量 Q
        key: [B, N, S, D] 前向键张量 K
        value: [B, N, S, D] 前向值张量 V
        dy: [B, N, S, D] 上游梯度 dO（对前向输出 O 的梯度）
        scaleValue: 缩放因子，通常为 1/sqrt(D)
        is_causal: 是否启用因果掩码（下三角，j <= i 可见），默认 False

    Returns:
        dq [B, N, S, D], dk [B, N, S, D], dv [B, N, S, D] — dtype 与输入一致
    """
    original_dtype = query.dtype
    dq, dk, dv = _flash_attention_backward_core(query, key, value, dy, scaleValue, is_causal, torch.float32)
    return dq.to(original_dtype), dk.to(original_dtype), dv.to(original_dtype)


def flash_attention_backward_oracle(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dy: torch.Tensor,
    scaleValue: float,
    is_causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Oracle (g)：dtype-agnostic，计算精度跟随输入（fp64_cpu 下为 fp64 真值），不硬编码 .float()。"""
    return _flash_attention_backward_core(query, key, value, dy, scaleValue, is_causal, query.dtype)
