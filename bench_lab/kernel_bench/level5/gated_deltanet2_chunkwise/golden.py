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
GatedDeltaNet2Chunkwise 算子 Torch Golden 参考实现

Gated DeltaNet-2（GDN-2）线性注意力层的前向：把 GDN-1 的标量 β 解耦成通道级 erase 门 b_t
（作用于 Dk 通道）与 write 门 w_t（作用于 Dv 通道），衰减 α_t = exp(g_t) 逐 Dk 通道。
Golden 采用与任意分块（chunkwise）算法数学等价的朴素逐 token 递归（每个 (b, h) 独立，
S_0 = 0 ∈ R^{Dk×Dv}）:
    q̂_t = q_t / (‖q_t‖₂ + 1e-6),  k̂_t = k_t / (‖k_t‖₂ + 1e-6)     （沿 Dk 做 L2 归一化）
    S̄_t = Diag(α_t) S_{t-1}                                        （按 Dk 行逐通道衰减）
    e_t = b_t ⊙ k̂_t,   r_t = S̄_tᵀ e_t ∈ R^{Dv}                       （通道级 erase 读出）
    z_t = w_t ⊙ v_t                                                 （通道级 write）
    S_t = S̄_t + k̂_t (z_t − r_t)ᵀ
    o_t = S_tᵀ q̂_t ∈ R^{Dv}
输出 y = [o_1, …, o_L]，final_state = S_L。
输入约定: erase_gate / write_gate ∈ (0, 1)，log_decay ≤ 0（均由 value_range 保证），
配合 k̂ 的 L2 归一化使 k̂ᵀ(b ⊙ k̂) ≤ 1、状态更新为收缩映射，任意随机输入下递归稳定。
chunk_size 仅约束 kernel 的分块实现，不影响数学结果。
plain golden 内部升 fp32 计算（bench 语义），输出转回输入 dtype；
gated_deltanet2_chunkwise_oracle 跟随输入精度（golden_precision=fp64_cpu 下即为 fp64 真值）。
"""


def _gated_deltanet2_chunkwise_core(q, k, v, erase_gate, write_gate, log_decay, compute_dtype):
    """核心计算：以 compute_dtype 精度执行朴素逐 token 递归，返回 (y, final_state)。"""
    Bsz, L, H, Dk = q.shape
    Dv = v.shape[-1]

    q_f = q.to(compute_dtype)
    k_f = k.to(compute_dtype)
    v_f = v.to(compute_dtype)
    b_f = erase_gate.to(compute_dtype)
    w_f = write_gate.to(compute_dtype)
    g_f = log_decay.to(compute_dtype)

    # 沿 Dk 做 L2 归一化（eps 加在范数上）
    q_hat = q_f / (q_f.norm(dim=-1, keepdim=True) + 1e-6)          # [B, L, H, Dk]
    k_hat = k_f / (k_f.norm(dim=-1, keepdim=True) + 1e-6)          # [B, L, H, Dk]
    alpha = torch.exp(g_f)                                          # [B, L, H, Dk] 通道级衰减 ∈ (0, 1]
    e_all = b_f * k_hat                                             # [B, L, H, Dk] erase 向量 e_t = b_t ⊙ k̂_t
    z_all = w_f * v_f                                               # [B, L, H, Dv] write 向量 z_t = w_t ⊙ v_t

    S = torch.zeros(Bsz, H, Dk, Dv, dtype=compute_dtype, device=q.device)
    y = torch.empty(Bsz, L, H, Dv, dtype=compute_dtype, device=q.device)
    for t in range(L):
        # S̄_t = Diag(α_t) S_{t-1}：按 Dk 行逐通道衰减
        S = alpha[:, t].unsqueeze(-1) * S                                          # [B, H, Dk, Dv]
        # r_t = S̄_tᵀ e_t（沿 Dk 内积）
        r = (S * e_all[:, t].unsqueeze(-1)).sum(dim=-2)                            # [B, H, Dv]
        # S_t = S̄_t + k̂_t (z_t − r_t)ᵀ（外积 [Dk] × [Dv]）
        S = S + k_hat[:, t].unsqueeze(-1) * (z_all[:, t] - r).unsqueeze(-2)        # [B, H, Dk, Dv]
        # o_t = S_tᵀ q̂_t（沿 Dk 内积）
        y[:, t] = (S * q_hat[:, t].unsqueeze(-1)).sum(dim=-2)                      # [B, H, Dv]
    return y, S


def gated_deltanet2_chunkwise(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    erase_gate: torch.Tensor,
    write_gate: torch.Tensor,
    log_decay: torch.Tensor,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gated DeltaNet-2 golden reference（朴素逐 token 递归；plain golden = bench：fp32 状态）

    Args:
        q: [B, L, H, Dk] 查询，算子内部沿 Dk 做 L2 归一化
        k: [B, L, H, Dk] 键，算子内部沿 Dk 做 L2 归一化
        v: [B, L, H, Dv] 值
        erase_gate: [B, L, H, Dk] 通道级 erase 门 b_t ∈ (0, 1)（评测取值范围 [0.05, 0.95]）
        write_gate: [B, L, H, Dv] 通道级 write 门 w_t ∈ (0, 1)（评测取值范围 [0.05, 0.95]）
        log_decay: [B, L, H, Dk] 通道级对数衰减 g_t ≤ 0（评测取值范围 [-0.5, -0.001]），α_t = exp(g_t)
        chunk_size: 分块大小，仅约束 kernel 分块实现，Golden 的朴素递归不使用

    Returns:
        y: [B, L, H, Dv] 输出序列，dtype 与 q 一致
        final_state: [B, H, Dk, Dv] 序列末尾的状态 S_L，dtype 与 q 一致
    """
    y, final_state = _gated_deltanet2_chunkwise_core(
        q, k, v, erase_gate, write_gate, log_decay, torch.float32)
    return y.to(q.dtype), final_state.to(q.dtype)


def gated_deltanet2_chunkwise_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    erase_gate: torch.Tensor,
    write_gate: torch.Tensor,
    log_decay: torch.Tensor,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Oracle (g)：dtype-agnostic，计算精度跟随输入（fp64_cpu 下为 fp64 真值），不硬编码 .float()。"""
    return _gated_deltanet2_chunkwise_core(q, k, v, erase_gate, write_gate, log_decay, q.dtype)
