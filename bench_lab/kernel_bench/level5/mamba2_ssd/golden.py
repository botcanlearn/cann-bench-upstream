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

"""
Mamba2Ssd 算子 Torch Golden 参考实现

Mamba2 分块状态空间对偶（SSD）。Golden 采用与 SSD 分块算法数学等价的朴素逐步递归：
    a_t = exp(dt_t · A)
    h_t = a_t · h_{t-1} + dt_t · (B_t ⊗ x_t)        (h_0 = 0，状态 [B, H, P, N])
    y_t = h_t · C_t + D_skip ⊙ x_t
B/C 按 KV 组 G 广播到 H 个头（H % G == 0，组内共享）。
输入约定: dt > 0 且 A < 0（由 value_range 保证），使 a_t ∈ (0, 1)、递归收缩稳定。
plain golden 内部升 fp32 计算（bench 语义），输出转回输入 dtype；
mamba2_ssd_oracle 跟随输入精度（golden_precision=fp64_cpu 下即为 fp64 真值）。
"""


def _mamba2_ssd_core(x, dt, A, B_mat, C_mat, D_skip, compute_dtype):
    """核心计算：以 compute_dtype 精度执行朴素逐步递归。"""
    Bsz, L, H, P = x.shape
    G = B_mat.shape[2]
    N = B_mat.shape[3]
    rep = H // G  # 组内共享：每组广播到 H/G 个头

    x_f = x.to(compute_dtype)
    dt_f = dt.to(compute_dtype)
    A_f = A.to(compute_dtype)
    D_f = D_skip.to(compute_dtype)
    # B/C 按组广播到 H 个头: [B, L, G, N] -> [B, L, H, N]
    B_h = B_mat.to(compute_dtype).repeat_interleave(rep, dim=2)
    C_h = C_mat.to(compute_dtype).repeat_interleave(rep, dim=2)

    # 循环外预计算逐步系数
    a = torch.exp(dt_f * A_f.view(1, 1, H))            # [B, L, H] 衰减系数
    dtx = dt_f.unsqueeze(-1) * x_f                     # [B, L, H, P] dt_t ⊙ x_t

    h = torch.zeros(Bsz, H, P, N, dtype=compute_dtype, device=x.device)
    y = torch.empty(Bsz, L, H, P, dtype=compute_dtype, device=x.device)
    for t in range(L):
        # h_t = a_t · h_{t-1} + dt_t · (B_t ⊗ x_t)，外积 [B,H,P,1] × [B,H,1,N]
        h = a[:, t].unsqueeze(-1).unsqueeze(-1) * h \
            + dtx[:, t].unsqueeze(-1) * B_h[:, t].unsqueeze(2)
        # y_t = h_t · C_t（沿 N 维内积）+ D_skip ⊙ x_t
        y[:, t] = (h * C_h[:, t].unsqueeze(2)).sum(dim=-1) \
            + D_f.view(1, H, 1) * x_f[:, t]
    return y


def mamba2_ssd(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B_mat: torch.Tensor,
    C_mat: torch.Tensor,
    D_skip: torch.Tensor,
    chunk_size: int = 256,
) -> torch.Tensor:
    """
    Mamba2 SSD golden reference（朴素递归；plain golden = bench：fp32 状态）

    Args:
        x: [B, L, H, P] 输入序列
        dt: [B, L, H] 逐 token 逐头时间步长 Δt（恒正）
        A: [H] 逐头状态转移标量（恒负）
        B_mat: [B, L, G, N] 输入投影系数，组内共享
        C_mat: [B, L, G, N] 状态读出系数，组内共享
        D_skip: [H] 逐头残差跳连系数
        chunk_size: SSD 分块大小，仅约束 kernel 分块实现，Golden 的朴素递归不使用

    Returns:
        y: [B, L, H, P] 输出序列，dtype 与 x 一致
    """
    y = _mamba2_ssd_core(x, dt, A, B_mat, C_mat, D_skip, torch.float32)
    return y.to(x.dtype)


def mamba2_ssd_oracle(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B_mat: torch.Tensor,
    C_mat: torch.Tensor,
    D_skip: torch.Tensor,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Oracle (g)：dtype-agnostic，计算精度跟随输入（fp64_cpu 下为 fp64 真值），不硬编码 .float()。"""
    return _mamba2_ssd_core(x, dt, A, B_mat, C_mat, D_skip, x.dtype)
