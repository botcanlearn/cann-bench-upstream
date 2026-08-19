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
BatchedSvd 算子 Torch Golden 参考实现

批量薄 SVD：A = U diag(S) V^T，U [B,M,N] 列正交，S [B,N] 降序非负，V [B,N,N]。
SVD 的奇异向量存在符号不唯一性（u_j, v_j 同时取反仍是合法分解），
为使逐元素比对有意义，golden 采用符号规范化：对每列 u_j 找 |u_j| 最大分量，
若为负则该列 u_j 与对应 v_j 同时取反。
plain golden 以 fp32 计算（bench 语义）；batched_svd_oracle 跟随输入精度
（golden_precision=fp64_cpu 下 torch.linalg.svd 以 fp64 计算，即为 fp64 真值）。
kernel 侧预期实现：单边 Jacobi 旋转迭代至收敛（off-diagonal 范数判据）。

钩子函数（框架自动调用，golden 与候选同等作用）：
  get_input  —— 把随机张量重建为良条件矩阵 A = Q_u · diag(σ) · Q_v^T，
                σ 为预设的良分离谱（线性 1.0 → 0.1，按各批次 max|a| 缩放），
                消除奇异值简并 / 接近 0 带来的 U、V 数值不适定。
  get_output —— 把 (u, s, v) 变换为结构化校验量后再逐元素比对：
                [U diag(S) V^T（重构）, S, cat(U^T(U⊙sgn), V^T V) + 1（正交性 + 符号约定）]，
                使 ||A − U diag(S) V^T||、U^T U ≈ I、V^T V ≈ I、S 排序/非负、
                符号规范化均在既有 checker 下被强制校验（对合法 SVD 的自由度不敏感）。
"""


def _batched_svd_core(a, compute_dtype):
    """核心计算：以 compute_dtype 精度执行薄 SVD + 符号规范化。"""
    a_f = a.to(compute_dtype)
    # torch.linalg.svd 返回 (U, S, Vh)，Vh = V^T；S 已按降序排列
    u, s, vh = torch.linalg.svd(a_f, full_matrices=False)   # [B,M,N], [B,N], [B,N,N]
    v = vh.transpose(-2, -1)                                # [B, N, N]，列向量形式

    # === 符号规范化（gather 定位）===
    # 对每列 u_j 找 |u_j| 最大分量的位置: idx [B, N]
    idx = u.abs().argmax(dim=-2)                            # 沿 M 维 argmax
    # gather 取出该分量的带符号值: pivot [B, N]
    pivot = u.gather(-2, idx.unsqueeze(-2)).squeeze(-2)
    # 若该分量为负，则该列 u_j 与对应 v_j 同时取反（pivot == 0 时不翻转）
    sign = torch.where(pivot < 0,
                       torch.tensor(-1.0, dtype=compute_dtype, device=a.device),
                       torch.tensor(1.0, dtype=compute_dtype, device=a.device))  # [B, N]
    u = u * sign.unsqueeze(-2)                              # 按列翻转
    v = v * sign.unsqueeze(-2)
    return u, s, v


def batched_svd(a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    批量薄 SVD golden reference（含符号规范化；plain golden = bench：fp32 计算）

    Args:
        a: [B, M, N] 待分解的批量矩阵，M ≥ N（由 cases 保证），float32

    Returns:
        u: [B, M, N] 左奇异向量（列正交，经符号规范化）
        s: [B, N] 奇异值，降序非负
        v: [B, N, N] 右奇异向量（列向量形式，A = U diag(S) V^T，经符号规范化）
    """
    original_dtype = a.dtype
    u, s, v = _batched_svd_core(a, torch.float32)
    return u.to(original_dtype), s.to(original_dtype), v.to(original_dtype)


def batched_svd_oracle(a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Oracle (g)：dtype-agnostic，计算精度跟随输入（fp64_cpu 下为 fp64 真值），不硬编码 .float()。"""
    return _batched_svd_core(a, a.dtype)


def _pivot_sign(u: torch.Tensor) -> torch.Tensor:
    """每列 |u_j| 最大分量的符号 [B, N]（符号规范化后恒为 +1；该分量幅值 ≥ 1/sqrt(M)，不为 0）。"""
    idx = u.abs().argmax(dim=-2)                                     # [B, N]
    pivot = u.gather(-2, idx.unsqueeze(-2)).squeeze(-2)              # [B, N]
    return torch.where(pivot < 0, -torch.ones_like(pivot), torch.ones_like(pivot))


def get_input(a: torch.Tensor, **kwargs) -> list:
    """把随机输入重建为良条件矩阵：A = Q_u · diag(σ) · Q_v^T（同时替换 golden 与候选的输入）。

    通用生成器给出的连续随机矩阵在 M≈N 时最小奇异值趋近 0、相邻奇异值可能几乎简并，
    U/V 对应列数值不适定，逐元素比对无意义。这里以 a 的 QR 正交因子作为 Q_u [B,M,N]、
    Q_v [B,N,N]，并施加预设的良分离谱 σ_j = c_b · (1 − 0.9·j/(N−1))（线性 1.0 → 0.1，
    c_b = max|a_b| 保留 value_range 的量级语义），使 cond(A) = 10、相邻谱间隔恒定。

    Returns:
        [a_conditioned]，dtype/shape 与 a 一致。
    """
    Bsz, M, N = a.shape
    a64 = a.to(torch.float64)
    q_u, _ = torch.linalg.qr(a64)                                        # [B, M, N] 列正交
    q_v, _ = torch.linalg.qr(a64[:, :N, :].transpose(-2, -1))            # [B, N, N] 列正交
    if N > 1:
        sigma = 1.0 - 0.9 * torch.arange(N, dtype=torch.float64, device=a.device) / (N - 1)
    else:
        sigma = torch.ones(1, dtype=torch.float64, device=a.device)
    scale = a64.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-3)   # [B, 1, 1]
    a_cond = (q_u * sigma.view(1, 1, N)) @ q_v.transpose(-2, -1) * scale # [B, M, N]
    return [a_cond.to(a.dtype)]


def get_output(u: torch.Tensor, s: torch.Tensor, v: torch.Tensor, **kwargs) -> list:
    """把 (u, s, v) 变换为对合法 SVD 自由度不敏感、但对结构性错误敏感的校验量。

    返回三个张量（对 golden / 候选 / 同精度参考统一变换后逐元素比对）：
      1. recon = U · diag(S) · V^T            [B, M, N]  —— 重构残差 ||A − U diag(S) V^T||
      2. s                                    [B, N]     —— 奇异值（golden 降序非负，逐元素比对即校验排序/非负）
      3. gram = cat(U^T (U ⊙ sgn), V^T V) + 1 [B, 2N, N] —— 正交性 U^T U ≈ I、V^T V ≈ I，
         其中 sgn = 每列 |u_j| 最大分量的符号：候选若未按约定规范化符号，对应对角元为 −1 → 不匹配；
         整体 +1 平移使 golden 无接近 0 的元素，正交性偏差按绝对量计入相对误差。
    例：候选 U_bad = 1.04·U（S、V 正确）→ recon ≈ 1.04·A、gram 对角 ≈ 2.0816 vs 2 → 被拒。
    """
    recon = (u * s.unsqueeze(-2)) @ v.transpose(-2, -1)                  # [B, M, N]
    u_signed = u * _pivot_sign(u).unsqueeze(-2)                          # 未规范化的列 → 取反
    gram_u = u.transpose(-2, -1) @ u_signed                              # [B, N, N]
    gram_v = v.transpose(-2, -1) @ v                                     # [B, N, N]
    gram = torch.cat([gram_u, gram_v], dim=-2) + 1.0                     # [B, 2N, N]
    return [recon, s, gram]
