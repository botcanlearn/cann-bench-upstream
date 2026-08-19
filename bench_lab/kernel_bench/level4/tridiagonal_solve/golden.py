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
TridiagonalSolve算子Torch Golden参考实现

批量三对角线性方程组求解 T @ x = b，T 由 (dl, d, du) 三条对角线给出：
    T[i, i-1] = dl[i], T[i, i] = d[i], T[i, i+1] = du[i]
plain golden 用向量化 Thomas 算法（前向消元 + 回代）以 fp32 计算（bench 语义），
不构造稠密矩阵；tridiagonal_solve_oracle 跟随输入精度
（golden_precision=fp64_cpu 下即为 fp64 真值）。
输入约定: d 严格对角占优 (|d| > |dl| + |du|，由 value_range 保证)，无需选主元。
kernel 侧预期采用 PCR/CR 并行消元。
"""


def _tridiagonal_solve_core(dl, d, du, b_rhs, compute_dtype):
    """核心计算：以 compute_dtype 精度执行 Thomas 前向消元 + 回代。"""
    dl_f = dl.to(compute_dtype)
    d_f = d.to(compute_dtype)
    du_f = du.to(compute_dtype)
    b_f = b_rhs.to(compute_dtype)
    Bsz, N, K = b_f.shape

    # 前向消元: c_prime [B, N], d_prime [B, N, K]
    c_prime = torch.zeros(Bsz, N, dtype=compute_dtype, device=b_rhs.device)
    d_prime = torch.zeros(Bsz, N, K, dtype=compute_dtype, device=b_rhs.device)
    c_prime[:, 0] = du_f[:, 0] / d_f[:, 0]
    d_prime[:, 0] = b_f[:, 0] / d_f[:, 0].unsqueeze(-1)
    for i in range(1, N):
        denom = d_f[:, i] - dl_f[:, i] * c_prime[:, i - 1]                     # [B]
        c_prime[:, i] = du_f[:, i] / denom                                     # 末行 c' 不参与回代
        d_prime[:, i] = (b_f[:, i] - dl_f[:, i].unsqueeze(-1) * d_prime[:, i - 1]) / denom.unsqueeze(-1)

    # 回代
    x = torch.zeros_like(d_prime)
    x[:, N - 1] = d_prime[:, N - 1]
    for i in range(N - 2, -1, -1):
        x[:, i] = d_prime[:, i] - c_prime[:, i].unsqueeze(-1) * x[:, i + 1]
    return x


def tridiagonal_solve(
    dl: torch.Tensor,
    d: torch.Tensor,
    du: torch.Tensor,
    b_rhs: torch.Tensor,
) -> torch.Tensor:
    """
    批量三对角线性方程组求解 (Thomas 算法；plain golden = bench：fp32 计算)

    Args:
        dl: [B, N] 下对角线, 首元素 dl[:, 0] 忽略, float32/float16
        d: [B, N] 主对角线 (严格对角占优), dtype 与 dl 一致
        du: [B, N] 上对角线, 末元素 du[:, N-1] 忽略, dtype 与 dl 一致
        b_rhs: [B, N, K] 右端项 (K 个右端向量共享同一系数矩阵), dtype 与 dl 一致

    Returns:
        x: [B, N, K] 解向量, 满足 T @ x = b_rhs, dtype 与输入一致
    """
    x = _tridiagonal_solve_core(dl, d, du, b_rhs, torch.float32)
    return x.to(b_rhs.dtype)


def tridiagonal_solve_oracle(
    dl: torch.Tensor,
    d: torch.Tensor,
    du: torch.Tensor,
    b_rhs: torch.Tensor,
) -> torch.Tensor:
    """Oracle (g)：dtype-agnostic，计算精度跟随输入（fp64_cpu 下为 fp64 真值），不硬编码 .float()。"""
    return _tridiagonal_solve_core(dl, d, du, b_rhs, b_rhs.dtype)
