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
StatevectorAdjointGradient 算子 Torch Golden 参考实现

量子态矢电路模拟 + 伴随法梯度（PennyLane-lightning / cuQuantum adjoint 的核心）。
复数以 [..., 2] 实部/虚部分离表示，全程实数算术（NPU 无复数类型）。

语义（详见 desc.md §2）:
  1) 归一化 psi <- state / (||state|| + 1e-12)（任意随机输入态合法）
  2) 电路 [G, 4] int32 逐行取模译码（任意非负整数合法）:
       gate = c0 mod 5 (0 H, 1 RZ, 2 RX, 3 CNOT, 4 CZ)
       qa = c1 mod n,  qb = c2 mod n,  tid = c3 mod T
     qa == qb 的两比特门（CNOT/CZ）定义为恒等门。
  3) 前向作用全部门得 |psi>；|phi> = O|psi>（O = ⊗_q P_q，pauli_obs 译码 0 I/1 X/2 Y/3 Z）；
     expval = Re<phi|psi>
  4) 伴随法逆序回放 g = G-1..0:
       psi <- U_g^dagger psi（回到门 g 之前的态）
       若 U_g 含参（RZ/RX）: grads[tid_g] += 2 * Re<phi| (dU_g/dtheta) |psi>
       phi <- U_g^dagger phi
     多条指令共享同一 tid 时梯度按链式法则累加。

plain golden 以 fp32 计算（bench 语义：复数模拟要求 fp32，不支持 fp16/bf16）；
statevector_adjoint_gradient_oracle 跟随输入精度（golden_precision=fp64_cpu 下即为
fp64 真值），不硬编码 .float()。
"""


def _apply_1q(psi, n, q, m):
    """对第 q 个 qubit 应用 2x2 复矩阵 m（python 复数元组 ((m00,m01),(m10,m11))）。

    psi: [2^n, 2]（最后一维 = 实部/虚部），返回新张量。
    视图 (2^(n-q-1), 2, 2^q, 2)：dim1 为 qubit q 的比特（跨步 2^q）。
    """
    v = psi.reshape(1 << (n - 1 - q), 2, 1 << q, 2)
    a_re, a_im = v[:, 0, :, 0], v[:, 0, :, 1]
    b_re, b_im = v[:, 1, :, 0], v[:, 1, :, 1]
    (m00, m01), (m10, m11) = m
    out = torch.empty_like(v)
    out[:, 0, :, 0] = m00.real * a_re - m00.imag * a_im + m01.real * b_re - m01.imag * b_im
    out[:, 0, :, 1] = m00.real * a_im + m00.imag * a_re + m01.real * b_im + m01.imag * b_re
    out[:, 1, :, 0] = m10.real * a_re - m10.imag * a_im + m11.real * b_re - m11.imag * b_im
    out[:, 1, :, 1] = m10.real * a_im + m10.imag * a_re + m11.real * b_im + m11.imag * b_re
    return out.reshape(psi.shape)


def _apply_cnot(psi, n, qa, qb):
    """CNOT：qa 为控制位，qb 为目标位（qa != qb 由调用方保证）。"""
    hi, lo = max(qa, qb), min(qa, qb)
    v = psi.reshape(1 << (n - 1 - hi), 2, 1 << (hi - 1 - lo), 2, 1 << lo, 2).clone()
    if qa == hi:                              # 控制位在 dim1，目标位在 dim3
        tmp = v[:, 1, :, 0].clone()
        v[:, 1, :, 0] = v[:, 1, :, 1]
        v[:, 1, :, 1] = tmp
    else:                                     # 控制位在 dim3，目标位在 dim1
        tmp = v[:, 0, :, 1].clone()
        v[:, 0, :, 1] = v[:, 1, :, 1]
        v[:, 1, :, 1] = tmp
    return v.reshape(psi.shape)


def _apply_cz(psi, n, qa, qb):
    """CZ：两比特均为 1 的振幅乘 -1（qa != qb 由调用方保证，qa/qb 对称）。"""
    hi, lo = max(qa, qb), min(qa, qb)
    v = psi.reshape(1 << (n - 1 - hi), 2, 1 << (hi - 1 - lo), 2, 1 << lo, 2).clone()
    v[:, 1, :, 1] = -v[:, 1, :, 1]
    return v.reshape(psi.shape)


_INV_SQRT2 = 1.0 / math.sqrt(2.0)
_H_MAT = ((complex(_INV_SQRT2, 0.0), complex(_INV_SQRT2, 0.0)),
          (complex(_INV_SQRT2, 0.0), complex(-_INV_SQRT2, 0.0)))


def _rz_mat(theta):
    h = 0.5 * theta
    return ((complex(math.cos(h), -math.sin(h)), complex(0.0, 0.0)),
            (complex(0.0, 0.0), complex(math.cos(h), math.sin(h))))


def _rx_mat(theta):
    h = 0.5 * theta
    return ((complex(math.cos(h), 0.0), complex(0.0, -math.sin(h))),
            (complex(0.0, -math.sin(h)), complex(math.cos(h), 0.0)))


def _drz_mat(theta):
    """dRZ/dtheta = (-i/2) diag(e^{-i theta/2}, -e^{i theta/2})。"""
    h = 0.5 * theta
    return ((complex(-0.5 * math.sin(h), -0.5 * math.cos(h)), complex(0.0, 0.0)),
            (complex(0.0, 0.0), complex(-0.5 * math.sin(h), 0.5 * math.cos(h))))


def _drx_mat(theta):
    """dRX/dtheta = 0.5 * [[-sin(h), -i cos(h)], [-i cos(h), -sin(h)]]，h = theta/2。"""
    h = 0.5 * theta
    return ((complex(-0.5 * math.sin(h), 0.0), complex(0.0, -0.5 * math.cos(h))),
            (complex(0.0, -0.5 * math.cos(h)), complex(-0.5 * math.sin(h), 0.0)))


def _apply_gate(psi, n, gate, qa, qb, theta):
    """按译码结果应用一个门；qa == qb 的两比特门为恒等门。"""
    if gate == 0:
        return _apply_1q(psi, n, qa, _H_MAT)
    if gate == 1:
        return _apply_1q(psi, n, qa, _rz_mat(theta))
    if gate == 2:
        return _apply_1q(psi, n, qa, _rx_mat(theta))
    if qa == qb:
        return psi
    if gate == 3:
        return _apply_cnot(psi, n, qa, qb)
    return _apply_cz(psi, n, qa, qb)


def _apply_gate_dagger(psi, n, gate, qa, qb, theta):
    """应用门的厄米共轭 U^dagger：H/CNOT/CZ 自伴，RZ/RX 取 -theta。"""
    if gate in (1, 2):
        return _apply_gate(psi, n, gate, qa, qb, -theta)
    return _apply_gate(psi, n, gate, qa, qb, theta)


def _apply_pauli(psi, n, pauli):
    """应用张量积可观测量 O = ⊗_q P_q（0 I、1 X、2 Y、3 Z），返回新张量。"""
    phi = psi
    for q in range(n):
        p = pauli[q]
        if p == 0:
            continue
        v = phi.reshape(1 << (n - 1 - q), 2, 1 << q, 2)
        out = torch.empty_like(v)
        if p == 1:                            # X: (a, b) -> (b, a)
            out[:, 0] = v[:, 1]
            out[:, 1] = v[:, 0]
        elif p == 2:                          # Y: (a, b) -> (-i*b, i*a)
            out[:, 0, :, 0] = v[:, 1, :, 1]
            out[:, 0, :, 1] = -v[:, 1, :, 0]
            out[:, 1, :, 0] = -v[:, 0, :, 1]
            out[:, 1, :, 1] = v[:, 0, :, 0]
        else:                                 # Z: (a, b) -> (a, -b)
            out[:, 0] = v[:, 0]
            out[:, 1] = -v[:, 1]
        phi = out.reshape(phi.shape)
    return phi


def _re_inner(x, y):
    """Re<x|y> = sum(x_re*y_re + x_im*y_im)。"""
    return (x[:, 0] * y[:, 0] + x[:, 1] * y[:, 1]).sum()


def _statevector_adjoint_gradient_core(state, circuit, thetas, pauli_obs, compute_dtype):
    """核心计算：归一化 -> 前向电路 -> 期望值 -> 伴随法逆序梯度。

    返回 (expval [1], grads [T])，精度为 compute_dtype。
    """
    dim = state.shape[0]
    n = dim.bit_length() - 1                  # dim = 2^n
    num_theta = thetas.shape[0]
    num_gates = circuit.shape[0]

    c = circuit.to(torch.int64)
    gate_v = (c[:, 0] % 5).tolist()
    qa_v = (c[:, 1] % n).tolist()
    qb_v = (c[:, 2] % n).tolist()
    tid_v = (c[:, 3] % num_theta).tolist()
    theta_v = [float(t) for t in thetas.to(compute_dtype).tolist()]
    pauli_v = (pauli_obs.to(torch.int64) % 4).tolist()

    # 1) 归一化（算子语义：任意随机输入态合法）
    psi = state.to(compute_dtype)
    norm = torch.sqrt((psi * psi).sum())
    psi = psi / (norm + 1e-12)

    # 2) 前向作用全部门
    for g in range(num_gates):
        psi = _apply_gate(psi, n, gate_v[g], qa_v[g], qb_v[g], theta_v[tid_v[g]])

    # 3) |phi> = O |psi>，expval = Re<phi|psi>
    phi = _apply_pauli(psi, n, pauli_v)
    expval = _re_inner(phi, psi).reshape(1)

    # 4) 伴随法逆序回放
    grads = torch.zeros(num_theta, dtype=compute_dtype, device=state.device)
    for g in range(num_gates - 1, -1, -1):
        gate, qa, qb, tid = gate_v[g], qa_v[g], qb_v[g], tid_v[g]
        theta = theta_v[tid]
        psi = _apply_gate_dagger(psi, n, gate, qa, qb, theta)   # 回到门 g 之前的态
        if gate == 1:
            dpsi = _apply_1q(psi, n, qa, _drz_mat(theta))
            grads[tid] = grads[tid] + 2.0 * _re_inner(phi, dpsi)
        elif gate == 2:
            dpsi = _apply_1q(psi, n, qa, _drx_mat(theta))
            grads[tid] = grads[tid] + 2.0 * _re_inner(phi, dpsi)
        phi = _apply_gate_dagger(phi, n, gate, qa, qb, theta)
    return expval, grads


def statevector_adjoint_gradient(
    state: torch.Tensor,
    circuit: torch.Tensor,
    thetas: torch.Tensor,
    pauli_obs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    量子态矢电路模拟 + 伴随法梯度 golden reference（plain golden = bench：fp32 计算）

    Args:
        state: [2^n, 2] float32 输入态（实部/虚部分离，算子内部归一化），n 由 shape 推出
        circuit: [G, 4] int32 电路指令，逐行取模译码（任意非负整数合法，
            评测取值范围 [0, 1073741823]）
        thetas: [T] float32 旋转门参数（评测取值范围 [-3.14159, 3.14159]），
            多条指令可共享同一参数（梯度累加）
        pauli_obs: [n] int32 张量积可观测量（0 I、1 X、2 Y、3 Z，评测取值范围 [0, 3]）

    Returns:
        expval: [1] float32 期望值 <psi_G|O|psi_G>
        grads: [T] float32 伴随法梯度 d expval / d thetas
    """
    expval, grads = _statevector_adjoint_gradient_core(
        state, circuit, thetas, pauli_obs, torch.float32)
    return expval.to(state.dtype), grads.to(state.dtype)


def statevector_adjoint_gradient_oracle(
    state: torch.Tensor,
    circuit: torch.Tensor,
    thetas: torch.Tensor,
    pauli_obs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Oracle (g)：dtype-agnostic，计算精度跟随 state（fp64_cpu 下为 fp64 真值），不硬编码 .float()。"""
    return _statevector_adjoint_gradient_core(
        state, circuit, thetas, pauli_obs, state.dtype)
