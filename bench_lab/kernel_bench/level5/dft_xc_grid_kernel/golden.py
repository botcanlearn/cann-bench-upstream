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

import math
from typing import Tuple

import torch

"""
DftXcGridKernel 算子 Torch Golden 参考实现

DFT 数值积分（numint）热点：格点密度评估 → PBE 交换-相关泛函逐点求值 → V_xc 矩阵组装。
无自旋极化（closed-shell）形式，全部公式与常数见 desc.md §2。

流水线（G 个格点，Nb 个基函数）:
  1. D = (dm + dmᵀ)/2（对称化投影，幂等）
     ρ_g = Σ_{μν} φ_gμ D_μν φ_gν，  (∇ρ)_g = 2 Σ_{μν} (∂φ)_gμ D_μν φ_gν（x/y/z 三分量），
     σ_g = |∇ρ_g|²
  2. 数值防护（规格的一部分，保证任意随机输入合法）:
     ρ ← max(ρ, 1e-8)；σ ← max(σ, 1e-20)；t² ← min(t², 1e8)。ρ 下限取 1e-8：保证 fp32
     通路上 s²/t² 的除式反向（分母平方 ≥ 6.9e-41）与 A²t⁴ 均不上溢/下溢出 inf；
     t² 上限处 H 已饱和于 −ε_c（相对偏差 ≤ 1/(A·1e8)）
  3. PBE 交换 + PW92/PBE 关联的逐点能量密度 e_xc(ρ, σ)（单位: Hartree/体积）
  4. exc_energy = Σ_g w_g e_xc,g（标量，shape [1]）
  5. v_ρ = ∂e_xc/∂ρ、v_σ = ∂e_xc/∂σ 为泛函的精确偏导（golden 用 torch.autograd 对
     ρ/σ 叶子张量求导；防护 max/min 在求导图内，截断处偏导为 0）
     vxc[μ,ν] = Σ_g w_g [ v_ρ φ_gμ φ_gν + 2 v_σ (∇ρ_g·(∇φ)_gμ φ_gν + φ_gμ ∇ρ_g·(∇φ)_gν) ]
     （对称矩阵；且满足变分一致性 ∂exc_energy/∂dm_μν = vxc_μν）
  6. 输出规范化（评测口径，规格的一部分，与 batched_svd 的符号规范化同型）:
     vxc_shifted = vxc + C，C = _VXC_SHIFT = 10.0。物理 V_xc 元素在过零处密集，
     相对误差判据在零点邻域无意义；常数平移使比较基准远离零点（逐元素判据变为
     |err|/(|vxc+C|) ≈ |err|/C 的绝对误差口径），能量阈值得以按化学精度收紧。
     使用方以 vxc_shifted − C 还原物理 V_xc。

plain golden 内部升 fp32 计算（bench 语义），输出转回输入 dtype；
dft_xc_grid_kernel_oracle 跟随输入精度（golden_precision=fp64_cpu 下即为 fp64 真值）。
"""

# === PBE 交换常数 ===
_PBE_KAPPA = 0.804
_PBE_MU = 0.2195149727645171

# === PBE 关联常数 ===
_PBE_BETA = 0.06672455060314922
_PBE_GAMMA = (1.0 - math.log(2.0)) / (math.pi ** 2)

# === PW92 无自旋极化 ε_c(r_s) 常数 ===
_PW92_A = 0.0310907
_PW92_ALPHA1 = 0.21370
_PW92_BETA1 = 7.5957
_PW92_BETA2 = 3.5876
_PW92_BETA3 = 1.6382
_PW92_BETA4 = 0.49294

# === 数值防护下限/上限（规格的一部分） ===
_RHO_FLOOR = 1e-8
_SIGMA_FLOOR = 1e-20
_T2_CEIL = 1e8

# === 输出规范化常数（规格的一部分，见 desc §2）===
# vxc_shifted = vxc + _VXC_SHIFT；使用方减去该常数还原物理 V_xc
_VXC_SHIFT = 10.0


def _xc_energy_density(rho_raw, sigma_raw):
    """逐点 XC 能量密度 e_xc(ρ, σ) [G]（含数值防护，可微，dtype 跟随输入）。"""
    rho = torch.clamp(rho_raw, min=_RHO_FLOOR)
    sigma = torch.clamp(sigma_raw, min=_SIGMA_FLOOR)

    # --- PBE 交换 ---
    # e_x^unif = −(3/4)(3/π)^{1/3} ρ^{4/3}
    ex_unif = -0.75 * (3.0 / math.pi) ** (1.0 / 3.0) * torch.pow(rho, 4.0 / 3.0)
    # s² = σ / (4 (3π²)^{2/3} ρ^{8/3})
    s2 = sigma / (4.0 * (3.0 * math.pi ** 2) ** (2.0 / 3.0) * torch.pow(rho, 8.0 / 3.0))
    # F_x = 1 + κ − κ/(1 + μ s²/κ)
    fx = 1.0 + _PBE_KAPPA - _PBE_KAPPA / (1.0 + _PBE_MU * s2 / _PBE_KAPPA)
    ex_density = ex_unif * fx

    # --- PW92 均匀电子气关联 ε_c^unif(r_s) ---
    # r_s = (3/(4πρ))^{1/3}
    rs = torch.pow(3.0 / (4.0 * math.pi) / rho, 1.0 / 3.0)
    rs_sqrt = torch.sqrt(rs)
    den = 2.0 * _PW92_A * (_PW92_BETA1 * rs_sqrt + _PW92_BETA2 * rs
                           + _PW92_BETA3 * rs * rs_sqrt + _PW92_BETA4 * rs * rs)
    eps_c = -2.0 * _PW92_A * (1.0 + _PW92_ALPHA1 * rs) * torch.log1p(1.0 / den)

    # --- PBE 梯度修正 H(ρ, σ, ε_c) ---
    # t² = σ / (4 k_s² ρ²)，k_s = √(4 k_F/π)，k_F = (3π²ρ)^{1/3}（无自旋极化 φ=1）
    kf = torch.pow(3.0 * math.pi ** 2 * rho, 1.0 / 3.0)
    t2 = sigma * math.pi / (16.0 * kf * rho * rho)
    t2 = torch.clamp(t2, max=_T2_CEIL)
    # A = (β/γ) / [exp(−ε_c/γ) − 1]
    a_h = (_PBE_BETA / _PBE_GAMMA) / torch.expm1(-eps_c / _PBE_GAMMA)
    at2 = a_h * t2
    # H = γ ln[1 + (β/γ) t² (1 + A t²)/(1 + A t² + A² t⁴)]
    h = _PBE_GAMMA * torch.log1p(
        (_PBE_BETA / _PBE_GAMMA) * t2 * (1.0 + at2) / (1.0 + at2 + at2 * at2))
    ec_density = rho * (eps_c + h)

    return ex_density + ec_density


def _dft_xc_grid_kernel_core(ao_values, ao_grad, dm, grid_weights, compute_dtype):
    """核心计算：以 compute_dtype 精度执行，返回 (exc_energy [1], vxc_matrix [Nb, Nb])。"""
    phi = ao_values.to(compute_dtype)
    dphi = ao_grad.to(compute_dtype)
    d = dm.to(compute_dtype)
    w = grid_weights.to(compute_dtype)

    # === 1. 密度矩阵对称化投影（幂等）+ 格点密度/梯度 ===
    d = (d + d.transpose(0, 1)) / 2
    phid = phi @ d                                             # [G, Nb] = (φ D)
    rho = (phi * phid).sum(dim=-1)                             # [G]
    grad_rho = 2.0 * (dphi * phid.unsqueeze(0)).sum(dim=-1)    # [3, G]
    sigma = (grad_rho * grad_rho).sum(dim=0)                   # [G]

    # === 2/3. 逐点泛函求值 + 精确偏导 v_ρ = ∂e/∂ρ、v_σ = ∂e/∂σ（autograd） ===
    rho_leaf = rho.detach().requires_grad_(True)
    sigma_leaf = sigma.detach().requires_grad_(True)
    e_xc = _xc_energy_density(rho_leaf, sigma_leaf)            # [G]
    vrho, vsigma = torch.autograd.grad(e_xc.sum(), (rho_leaf, sigma_leaf))

    # === 4. 能量积分 ===
    exc_energy = (w * e_xc.detach()).sum().reshape(1)          # [1]

    # === 5. V_xc 矩阵组装（GEMM） ===
    wvr = w * vrho                                             # [G]
    wvs = w * vsigma                                           # [G]
    m = (grad_rho.unsqueeze(-1) * dphi).sum(dim=0)             # [G, Nb] = ∇ρ·∇φ
    vxc = phi.transpose(0, 1) @ (wvr.unsqueeze(-1) * phi)
    vxc = vxc + 2.0 * (m.transpose(0, 1) @ (wvs.unsqueeze(-1) * phi)
                       + phi.transpose(0, 1) @ (wvs.unsqueeze(-1) * m))

    # === 6. 输出规范化：常数平移（评测口径，规格的一部分，见 desc §2）===
    vxc_shifted = vxc + _VXC_SHIFT
    return exc_energy.contiguous(), vxc_shifted.contiguous()


def dft_xc_grid_kernel(
    ao_values: torch.Tensor,
    ao_grad: torch.Tensor,
    dm: torch.Tensor,
    grid_weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DFT XC 数值积分 golden reference（plain golden = bench：fp32 数据通路）

    Args:
        ao_values: [G, Nb] 基函数在格点上的值 φ_gμ
        ao_grad: [3, G, Nb] 基函数梯度三分量 (∂φ/∂x, ∂φ/∂y, ∂φ/∂z)
        dm: [Nb, Nb] 密度矩阵，算子先对称化 D = (dm + dmᵀ)/2
        grid_weights: [G] 积分权重（正，评测取值范围 [1e-6, 1e-2]）

    Returns:
        exc_energy: [1] XC 能量 Σ_g w_g e_xc,g，dtype 与 ao_values 一致
        vxc_shifted: [Nb, Nb] 平移后 XC 势矩阵 V_xc + 10.0（对称；
                     ∂exc_energy/∂dm_μν = vxc_shifted_μν − 10.0），dtype 与 ao_values 一致
    """
    exc_energy, vxc_shifted = _dft_xc_grid_kernel_core(
        ao_values, ao_grad, dm, grid_weights, torch.float32)
    return exc_energy.to(ao_values.dtype), vxc_shifted.to(ao_values.dtype)


def dft_xc_grid_kernel_oracle(
    ao_values: torch.Tensor,
    ao_grad: torch.Tensor,
    dm: torch.Tensor,
    grid_weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Oracle (g)：dtype-agnostic，计算精度跟随输入（fp64_cpu 下为 fp64 真值），不硬编码 .float()。"""
    return _dft_xc_grid_kernel_core(
        ao_values, ao_grad, dm, grid_weights, ao_values.dtype)
