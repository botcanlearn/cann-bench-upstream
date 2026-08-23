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
MultigridVcycle 算子 Torch Golden 参考实现

3D Poisson 方程 −Δu = f（零 Dirichlet 边界）的一次几何多重网格 V-cycle。每个 batch 独立。
网格含边界点（N = 2^k + 1，内点语义），u 的最外层在所有操作后保持 0；置零是算子语义的
一部分：进入 V-cycle 前先将 u0 的最外层置 0（任意随机输入合法），f 的边界值不参与计算。

- 算子：      (A u)[i,j,k] = (6*u[i,j,k] − Σ 六邻居) / h²        （仅内点）
- 平滑：      红黑 Gauss–Seidel。红点 = 全局 0-based 索引 (i+j+k) 偶，先全部红点、
              再全部黑点；同色点互不相邻 ⇒ 同色内并行顺序无关，红黑先后即规格。
              更新式 u[i,j,k] = (h²*f[i,j,k] + Σ 六邻居) / 6
- 残差：      r = f − A u（边界为 0）
- 限制：      full-weighting 27 点核 (1/64)*[1,2,1]⊗[1,2,1]⊗[1,2,1]，
              粗点 (I,J,K) 对应细点 (2I,2J,2K)，粗网格 Nc = (N−1)/2 + 1，h ← 2h
- 延拓：      三线性插值（偶/奇索引 8 种权重情形），粗边界为 0 ⇒ 细边界自动保持 0
- V-cycle：   pre_smooth 次平滑 → 残差 → 限制 → 递归求误差方程（初值 0）→
              延拓修正 u ← u + P(e) → post_smooth 次平滑；
              最粗层（num_levels 次粗化后）以 20 次红黑平滑代替直接解。

num_levels = 粗化（限制）次数，网格层数 = num_levels + 1；case 保证最粗层 ≥ 5³
（即 (N−1) / 2^num_levels ≥ 4）。

plain golden 内部升 fp32 计算（bench 语义），输出转回输入 dtype；
multigrid_vcycle_oracle 跟随输入精度（golden_precision=fp64_cpu 下即为 fp64 真值）。
"""

# 27 点 full-weighting 权重（[1,2,1] 张量积 / 64），以偏移 (a,b,c) ∈ {-1,0,1}³ 索引
_FW_W1 = (1.0, 2.0, 1.0)


def _redblack_masks(n, device):
    """内点红/黑掩码（红 = 全局 0-based 索引 i+j+k 偶），shape [n-2, n-2, n-2]。"""
    idx = torch.arange(1, n - 1, device=device)
    par = (idx.view(-1, 1, 1) + idx.view(1, -1, 1) + idx.view(1, 1, -1)) % 2
    red = par == 0
    return red, ~red


def _smooth_redblack(u, f, h2, sweeps, masks):
    """红黑 Gauss–Seidel：每个 sweep 先全部红点再全部黑点（就地更新 u 内点）。"""
    red, black = masks
    for _ in range(sweeps):
        for mask in (red, black):
            nb = (u[:, :-2, 1:-1, 1:-1] + u[:, 2:, 1:-1, 1:-1]
                  + u[:, 1:-1, :-2, 1:-1] + u[:, 1:-1, 2:, 1:-1]
                  + u[:, 1:-1, 1:-1, :-2] + u[:, 1:-1, 1:-1, 2:])
            upd = (h2 * f[:, 1:-1, 1:-1, 1:-1] + nb) / 6.0
            inner = u[:, 1:-1, 1:-1, 1:-1]
            u[:, 1:-1, 1:-1, 1:-1] = torch.where(mask, upd, inner)
    return u


def _residual(u, f, h2):
    """r = f − A u（内点），边界置 0。"""
    r = torch.zeros_like(u)
    nb = (u[:, :-2, 1:-1, 1:-1] + u[:, 2:, 1:-1, 1:-1]
          + u[:, 1:-1, :-2, 1:-1] + u[:, 1:-1, 2:, 1:-1]
          + u[:, 1:-1, 1:-1, :-2] + u[:, 1:-1, 1:-1, 2:])
    au = (6.0 * u[:, 1:-1, 1:-1, 1:-1] - nb) / h2
    r[:, 1:-1, 1:-1, 1:-1] = f[:, 1:-1, 1:-1, 1:-1] - au
    return r


def _restrict_fw(r):
    """full-weighting 限制：粗内点 (I,J,K) ← (1/64) Σ w_a w_b w_c * r[2I+a, 2J+b, 2K+c]。"""
    n = r.shape[-1]
    nc = (n - 1) // 2 + 1
    rc = torch.zeros(r.shape[0], nc, nc, nc, dtype=r.dtype, device=r.device)
    acc = torch.zeros(r.shape[0], nc - 2, nc - 2, nc - 2, dtype=r.dtype, device=r.device)
    # 粗内点 I=1..nc-2 对应细点 2I=2..n-3；偏移窗口切片 [2+a : n-3+a+1 : 2]
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            for c in (-1, 0, 1):
                w = _FW_W1[a + 1] * _FW_W1[b + 1] * _FW_W1[c + 1]
                acc = acc + w * r[:, 2 + a: n - 2 + a: 2,
                                  2 + b: n - 2 + b: 2,
                                  2 + c: n - 2 + c: 2]
    rc[:, 1:-1, 1:-1, 1:-1] = acc / 64.0
    return rc


def _prolong_trilinear(ec, n):
    """三线性延拓：粗 [B,nc,nc,nc] → 细 [B,n,n,n]（粗边界为 0 ⇒ 细边界为 0）。"""
    ef = torch.zeros(ec.shape[0], n, n, n, dtype=ec.dtype, device=ec.device)
    # 三偶：直接注入
    ef[:, ::2, ::2, ::2] = ec
    # 单奇（3 种）：沿奇方向两点平均
    ef[:, 1::2, ::2, ::2] = 0.5 * (ec[:, :-1, :, :] + ec[:, 1:, :, :])
    ef[:, ::2, 1::2, ::2] = 0.5 * (ec[:, :, :-1, :] + ec[:, :, 1:, :])
    ef[:, ::2, ::2, 1::2] = 0.5 * (ec[:, :, :, :-1] + ec[:, :, :, 1:])
    # 双奇（3 种）：面上四点平均
    ef[:, 1::2, 1::2, ::2] = 0.25 * (ec[:, :-1, :-1, :] + ec[:, :-1, 1:, :]
                                     + ec[:, 1:, :-1, :] + ec[:, 1:, 1:, :])
    ef[:, 1::2, ::2, 1::2] = 0.25 * (ec[:, :-1, :, :-1] + ec[:, :-1, :, 1:]
                                     + ec[:, 1:, :, :-1] + ec[:, 1:, :, 1:])
    ef[:, ::2, 1::2, 1::2] = 0.25 * (ec[:, :, :-1, :-1] + ec[:, :, :-1, 1:]
                                     + ec[:, :, 1:, :-1] + ec[:, :, 1:, 1:])
    # 三奇：体上八点平均
    ef[:, 1::2, 1::2, 1::2] = 0.125 * (
        ec[:, :-1, :-1, :-1] + ec[:, :-1, :-1, 1:] + ec[:, :-1, 1:, :-1] + ec[:, :-1, 1:, 1:]
        + ec[:, 1:, :-1, :-1] + ec[:, 1:, :-1, 1:] + ec[:, 1:, 1:, :-1] + ec[:, 1:, 1:, 1:])
    return ef


_COARSEST_SWEEPS = 20


def _vcycle(u, f, h, coarsenings_left, pre, post, mask_cache):
    """递归 V-cycle（u 就地更新并返回）。coarsenings_left = 0 表示已在最粗层。"""
    n = u.shape[-1]
    h2 = h * h
    if n not in mask_cache:
        mask_cache[n] = _redblack_masks(n, u.device)
    masks = mask_cache[n]

    if coarsenings_left == 0:
        return _smooth_redblack(u, f, h2, _COARSEST_SWEEPS, masks)

    u = _smooth_redblack(u, f, h2, pre, masks)
    r = _residual(u, f, h2)
    rc = _restrict_fw(r)
    ec = torch.zeros_like(rc)
    ec = _vcycle(ec, rc, 2.0 * h, coarsenings_left - 1, pre, post, mask_cache)
    u = u + _prolong_trilinear(ec, n)
    u = _smooth_redblack(u, f, h2, post, masks)
    return u


def _multigrid_vcycle_core(u0, f, num_levels, pre_smooth, post_smooth, h, compute_dtype):
    """核心计算：以 compute_dtype 精度执行一次 V-cycle，返回 u_out。"""
    u = u0.to(compute_dtype).clone()
    # 边界置 0 是算子语义的一部分（任意随机 u0 输入合法）
    u[:, 0, :, :] = 0
    u[:, -1, :, :] = 0
    u[:, :, 0, :] = 0
    u[:, :, -1, :] = 0
    u[:, :, :, 0] = 0
    u[:, :, :, -1] = 0
    f_c = f.to(compute_dtype)
    u = _vcycle(u, f_c, float(h), int(num_levels),
                int(pre_smooth), int(post_smooth), {})
    return u.contiguous()


def multigrid_vcycle(
    u0: torch.Tensor,
    f: torch.Tensor,
    num_levels: int,
    pre_smooth: int,
    post_smooth: int,
    h: float,
) -> torch.Tensor:
    """
    3D Poisson 几何多重网格 V-cycle golden reference（plain golden = bench：fp32 计算）

    Args:
        u0: [B, N, N, N] 初始猜测（N = 2^k + 1；算子先将最外层置 0）
        f: [B, N, N, N] 右端项 −Δu = f（f 的边界值不参与计算）
        num_levels: 粗化次数（评测取值 2 ~ 5；case 保证最粗层 ≥ 5³）
        pre_smooth: 前平滑红黑 sweep 数（评测取值 1 ~ 4）
        post_smooth: 后平滑红黑 sweep 数（评测取值 1 ~ 4）
        h: 最细层网格间距（评测取值范围 [0.01, 1.0]；粗一层 h 翻倍）

    Returns:
        u_out: [B, N, N, N] 一次 V-cycle 后的解，dtype 与 u0 一致，最外层为 0
    """
    u_out = _multigrid_vcycle_core(
        u0, f, num_levels, pre_smooth, post_smooth, h, torch.float32)
    return u_out.to(u0.dtype)


def multigrid_vcycle_oracle(
    u0: torch.Tensor,
    f: torch.Tensor,
    num_levels: int,
    pre_smooth: int,
    post_smooth: int,
    h: float,
) -> torch.Tensor:
    """Oracle (g)：dtype-agnostic，计算精度跟随输入（fp64_cpu 下为 fp64 真值），不硬编码 .float()。"""
    return _multigrid_vcycle_core(
        u0, f, num_levels, pre_smooth, post_smooth, h, u0.dtype)
