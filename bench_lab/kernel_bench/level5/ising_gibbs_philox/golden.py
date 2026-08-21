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
IsingGibbsPhilox 算子 Torch Golden 参考实现

周期边界 2D 晶格上的棋盘（checkerboard）Gibbs 更新，随机数由计数器型 RNG
Philox4x32-10 按 (sweep, color, batch, site) 计数器即时生成——bit-exact 随机模拟，
输出 int8 零容差。accept_table 是任意 [2, 5] 的 32 位无符号阈值表（int64 承载），
它定义一个确定性元胞自动机；物理 Boltzmann 表只是特例，评测不要求物理性。

一个 sweep = 先更新全部黑格（i+j 偶），再更新全部白格（i+j 奇）。同色格子的四邻居
全是异色，相互独立，更新顺序无关（并行顺序即规格）。对格点 (b, i, j)、sweep s、
颜色相 c（黑 0 白 1）:
    nsum = Σ 四邻居 σ，σ = 2·spin − 1 ∈ {−1, +1}，邻居按周期边界取
    k = (nsum + 4) / 2 ∈ {0..4}
    r = Philox4x32-10(key=(seed_hi, seed_lo), counter=(s, c, b, i·W+j)).x0
    若 r < accept_table[spin, k] 则翻转 spin ← 1 − spin
同一 sweep 内白格看到的是更新后的黑格。重复 num_sweeps 次。

Philox4x32-10（Random123 规范）: counter 4×32 位字 (c0,c1,c2,c3) = (s, c, b, i·W+j)，
key (k0,k1) = (seed_hi, seed_lo)。每轮:
    (c0,c1,c2,c3) ← (mulhi(M1,c2)^c1^k0, mullo(M1,c2), mulhi(M0,c0)^c3^k1, mullo(M0,c0))
    k0 ← (k0 + 0x9E3779B9) mod 2^32,  k1 ← (k1 + 0xBB67AE85) mod 2^32
共 10 轮，M0 = 0xD2511F53、M1 = 0xCD9E8D57，mullo/mulhi 为 32×32→64 位乘积的
低/高 32 位。输出取第一个字 x0（轮末的 c0）。

全整数 torch 实现：spins 以 int64 参与中间计算，输出转回 int8。32×32 位乘积最大
可达 (2^32−1)^2 ≈ 1.8e19，超出带符号 int64 上界 2^63−1 ≈ 9.2e18，因此 mullo/mulhi
以 16 位拆分实现（全部中间量 < 2^49，int64 精确无溢出）。整数精确运算下 plain 与
oracle 结果完全一致，oracle 直接复用同一核心。每个颜色相内对全部同色格子一次
向量化更新（同色独立性保证与任意更新顺序等价）。
"""

_M0 = 0xD2511F53
_M1 = 0xCD9E8D57
_W0 = 0x9E3779B9
_W1 = 0xBB67AE85
_MASK32 = 0xFFFFFFFF


def _mulhilo32(m, x):
    """32×32 位乘积的 (hi, lo) 32 位字。m 为 python int 常数，x 为 int64 张量（值 < 2^32）。

    直接乘会超 int64（(2^32−1)^2 > 2^63−1），按 16 位拆分：
    m·x = m·x_hi·2^16 + m·x_lo，全部中间量 < 2^49。
    """
    x_hi = x >> 16
    x_lo = x & 0xFFFF
    p_hi = m * x_hi                      # < 2^48
    p_lo = m * x_lo                      # < 2^48
    t = p_lo + ((p_hi & 0xFFFF) << 16)   # < 2^49
    lo = t & _MASK32
    hi = (p_hi >> 16) + (t >> 32)        # < 2^32
    return hi, lo


def _philox4x32_10_x0(c0, c1, c2, c3, k0, k1):
    """Philox4x32-10，返回输出第一个字 x0。c0..c3 为 int64 张量（可广播），k0/k1 为 python int。"""
    for _ in range(10):
        hi0, lo0 = _mulhilo32(_M0, c0)
        hi1, lo1 = _mulhilo32(_M1, c2)
        c0 = hi1 ^ c1 ^ k0
        c1 = lo1
        c2 = hi0 ^ c3 ^ k1
        c3 = lo0
        k0 = (k0 + _W0) & _MASK32
        k1 = (k1 + _W1) & _MASK32
    return c0


def _ising_gibbs_philox_core(spins, accept_table, num_sweeps, seed_hi, seed_lo):
    """核心计算：全整数棋盘 Gibbs 更新，返回 int8 spins_out。"""
    Bsz, H, W = spins.shape
    if H % 2 != 0 or W % 2 != 0:
        raise ValueError(f"H and W must be even for checkerboard 2-coloring, got {H}x{W}")
    dev = spins.device
    seed_hi = int(seed_hi) & _MASK32
    seed_lo = int(seed_lo) & _MASK32

    s_flat = spins.to(torch.int64).reshape(Bsz, H * W)      # [B, HW]，值 ∈ {0, 1}
    table = accept_table.to(torch.int64).reshape(10)        # 展平 [2, 5] → spin*5 + k

    ii = torch.arange(H, dtype=torch.int64, device=dev).view(H, 1).expand(H, W)
    jj = torch.arange(W, dtype=torch.int64, device=dev).view(1, W).expand(H, W)
    site = (ii * W + jj).reshape(H * W)                     # 计数器字 c3 = i·W + j
    # 周期边界四邻居的扁平下标 [HW, 4]（上、下、左、右）
    nb = torch.stack([
        ((ii - 1) % H) * W + jj,
        ((ii + 1) % H) * W + jj,
        ii * W + (jj - 1) % W,
        ii * W + (jj + 1) % W,
    ], dim=-1).reshape(H * W, 4)

    color = ((ii + jj) % 2).reshape(H * W)
    b_idx = torch.arange(Bsz, dtype=torch.int64, device=dev).view(Bsz, 1)

    per_color = []
    for c in (0, 1):
        sites_c = torch.nonzero(color == c, as_tuple=False).reshape(-1)   # [HW/2]
        per_color.append((sites_c, nb[sites_c], site[sites_c]))

    for s in range(int(num_sweeps)):
        for c in (0, 1):
            sites_c, nb_c, ctr3 = per_color[c]
            sigma = 2 * s_flat - 1                                        # [B, HW]
            nsum = sigma[:, nb_c].sum(dim=-1)                             # [B, HW/2]
            k = (nsum + 4) >> 1                                           # ∈ {0..4}
            cur = s_flat[:, sites_c]                                      # [B, HW/2]
            thr = table[cur * 5 + k]                                      # [B, HW/2]
            r = _philox4x32_10_x0(
                torch.tensor(s, dtype=torch.int64, device=dev),
                torch.tensor(c, dtype=torch.int64, device=dev),
                b_idx, ctr3.unsqueeze(0), seed_hi, seed_lo)               # [B, HW/2]
            s_flat[:, sites_c] = torch.where(r < thr, 1 - cur, cur)
    return s_flat.reshape(Bsz, H, W).to(torch.int8)


def ising_gibbs_philox(
    spins: torch.Tensor,
    accept_table: torch.Tensor,
    num_sweeps: int,
    seed_hi: int,
    seed_lo: int,
) -> torch.Tensor:
    """
    棋盘 Gibbs + Philox4x32-10 golden reference（全整数精确运算，输出零容差）

    Args:
        spins: [B, H, W] int8，取值 {0, 1}（0 代表自旋 −1，1 代表 +1）；H、W 均为偶数
        accept_table: [2, 5] int64 阈值表，取值 [0, 2^32−1]，行 = 当前 spin，列 = 邻居和
            索引 k = (nsum+4)/2；任意随机表均合法（定义一个确定性元胞自动机）
        num_sweeps: sweep 次数（一个 sweep = 黑格相 + 白格相）
        seed_hi: Philox key 高 32 位字（按无符号 32 位解释）
        seed_lo: Philox key 低 32 位字（按无符号 32 位解释）

    Returns:
        spins_out: [B, H, W] int8，num_sweeps 个 sweep 后的自旋场（零容差精确比对）
    """
    return _ising_gibbs_philox_core(spins, accept_table, num_sweeps, seed_hi, seed_lo)


def ising_gibbs_philox_oracle(
    spins: torch.Tensor,
    accept_table: torch.Tensor,
    num_sweeps: int,
    seed_hi: int,
    seed_lo: int,
) -> torch.Tensor:
    """Oracle (g)：整数域精确运算，与 plain golden 完全一致，直接复用核心。"""
    return _ising_gibbs_philox_core(spins, accept_table, num_sweeps, seed_hi, seed_lo)
