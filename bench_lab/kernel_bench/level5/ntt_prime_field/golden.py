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
NttPrimeField 算子 Torch Golden 参考实现

素域 Z_p 上的批量数论变换（Number Theoretic Transform）。所有运算为模 p 整数
运算，结果唯一确定，输出零容差（int32 阈值 0，逐 bit 完全相等）。

公式（ω = g^((p-1)/N) mod p，g 为 p 的原根，N 为 2 的幂且 N | (p-1)）:
    前向:   y_j = Σ_{i=0}^{N-1} x_i · ω^{ij}   mod p
    逆变换: y_j = N^{-1} · Σ_{i=0}^{N-1} x_i · ω^{-ij}   mod p   （N^{-1} 为 N 的模逆）
    输入输出均为自然顺序（不要求位逆序）。

golden 为向量化的迭代 radix-2 Cooley–Tukey（DIT）：位逆序置换 + log2(N) 个蝶形
阶段，全程 int64 运算、每次乘法后取模（p < 2^31，乘积 < 2^62 不溢出），不依赖
浮点。整数精确运算下 plain 与 oracle 结果完全一致，oracle 直接复用同一核心。
"""

# 内置素数 → 原根表（均满足 p < 2^31，p-1 含大 2-幂因子）
_PRIMITIVE_ROOTS = {
    2013265921: 31,   # BabyBear: 15 · 2^27 + 1，zk-STARK 常用域
    998244353: 3,     # 119 · 2^23 + 1，竞赛/多项式乘法常用
    167772161: 3,     # 5 · 2^25 + 1
    469762049: 3,     # 7 · 2^26 + 1
    754974721: 11,    # 45 · 2^24 + 1
}


def _ntt_prime_field_core(x, modulus, inverse):
    """核心计算：向量化迭代 radix-2 Cooley–Tukey（DIT），int64 模运算。"""
    p = int(modulus)
    if p not in _PRIMITIVE_ROOTS:
        raise ValueError(f"unsupported modulus {p}, expected one of {sorted(_PRIMITIVE_ROOTS)}")
    Bsz, N = x.shape
    if N & (N - 1) != 0:
        raise ValueError(f"N must be a power of two, got {N}")
    if (p - 1) % N != 0:
        raise ValueError(f"N={N} does not divide p-1={p - 1}")

    logn = N.bit_length() - 1
    v = x.to(torch.int64)

    # 主 N 次单位根 ω = g^((p-1)/N)；逆变换用 ω^{-1} = ω^{p-2}（费马小定理）
    omega = pow(_PRIMITIVE_ROOTS[p], (p - 1) // N, p)
    if inverse:
        omega = pow(omega, p - 2, p)

    if N > 1:
        idx = torch.arange(N, dtype=torch.int64, device=x.device)

        # 位逆序置换（DIT 输入序），log2(N) 次向量化位操作
        rev = torch.zeros_like(idx)
        for i in range(logn):
            rev = (rev << 1) | ((idx >> i) & 1)
        v = v[:, rev]

        # 全局 twiddle 表 tw[k] = ω^k (0 ≤ k < N/2)，向量化模幂（对指数位循环）
        e = idx[: N // 2]
        tw = torch.ones(N // 2, dtype=torch.int64, device=x.device)
        base = omega % p
        for i in range(logn):
            tw = torch.where((e >> i) & 1 == 1, (tw * base) % p, tw)
            base = (base * base) % p

        # log2(N) 个蝶形阶段：阶段 s 的块大小 m = 2^s，块内 twiddle 为 (ω^{N/m})^k = tw[k·N/m]
        for s in range(1, logn + 1):
            m = 1 << s
            half = m >> 1
            w = tw[:: N // m][:half]                        # [half]
            blocks = v.reshape(Bsz, N // m, m)
            u = blocks[:, :, :half]                         # [B, N/m, half]
            t = (blocks[:, :, half:] * w) % p               # 乘积 < p^2 < 2^62，int64 无溢出
            v = torch.cat([(u + t) % p, (u - t) % p], dim=-1).reshape(Bsz, N)

    if inverse:
        n_inv = pow(N, p - 2, p)                            # N^{-1} mod p
        v = (v * n_inv) % p

    return v.to(torch.int32)


def ntt_prime_field(
    x: torch.Tensor,
    modulus: int = 2013265921,
    inverse: bool = False,
) -> torch.Tensor:
    """
    素域 NTT golden reference（整数精确运算，零容差）

    Args:
        x: [B, N] int32 输入，取值 [0, modulus-1]；N 为 2 的幂且 N | (modulus-1)
        modulus: NTT 友好素数 p（内置原根表支持 2013265921 / 998244353 /
            167772161 / 469762049 / 754974721，均 < 2^31），默认 BabyBear
        inverse: False 为前向变换，True 为逆变换（结果乘 N^{-1} mod p）

    Returns:
        y: [B, N] int32，取值 [0, modulus-1]，自然顺序
    """
    return _ntt_prime_field_core(x, modulus, inverse)


def ntt_prime_field_oracle(
    x: torch.Tensor,
    modulus: int = 2013265921,
    inverse: bool = False,
) -> torch.Tensor:
    """Oracle (g)：整数域精确运算，与 plain golden 完全一致，直接复用核心。"""
    return _ntt_prime_field_core(x, modulus, inverse)
