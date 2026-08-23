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
RtmWaveStencil 算子 Torch Golden 参考实现

地震逆时偏移（RTM）正/反向传播的核心 kernel：二维声波方程的 8 阶空间中心差分
显式时间步进（leapfrog），叠加简化 Cerjan 型海绵吸收边界。每个 batch 独立。

单步更新（8 阶标准 FD 系数 c0=-205/72, c1=8/5, c2=-1/5, c3=8/315, c4=-1/560，
已对解析函数数值验证 8 阶收敛）：
    lap[i,j] = ( 2*c0*p[i,j] + Σ_{k=1..4} c_k * ((p[i-k,j]+p[i+k,j]) + (p[i,j-k]+p[i,j+k])) ) / dx²
    p_next = (2*p_curr − p_prev + (v*dt)² * lap) * g
    g[i,j] = 1 / (1 + damp_z[i] + damp_x[j])          （海绵衰减，g ≤ 1）
边界：固定零边界——p 的最外 4 圈（stencil halo 宽度）恒为 0。置零是算子语义的
一部分：第一步前先把 p_prev/p_curr 的最外 4 圈置 0（任意随机输入合法），此后
每步更新完成后再把 p_next 的最外 4 圈置 0。
滚动：p_prev ← p_curr，p_curr ← p_next；重复 num_steps 次，输出最终两个时间片。

数值有界性由 case 生成保证 CFL 约束 v_max*dt/dx ≤ 0.4（本格式 2D 稳定上限
约 0.5546 = sqrt(4/13.0032)，13.0032 为 8 阶叉形 stencil 的 2D 谱半径系数），
配合 g ≤ 1 使任意满足 CFL 的随机输入下 64 步内场值有界。

plain golden 内部升 fp32 计算（bench 语义），输出转回输入 dtype；
rtm_wave_stencil_oracle 跟随输入精度（golden_precision=fp64_cpu 下即为 fp64 真值）。
"""

# 8 阶中心差分二阶导系数（标准 Taylor 展开系数，见 desc §2 表格；已数值验证 8 阶收敛）
_FD8_C0 = -205.0 / 72.0
_FD8_CK = (8.0 / 5.0, -1.0 / 5.0, 8.0 / 315.0, -1.0 / 560.0)
_HALO = 4


def _zero_halo(p):
    """将最外 4 圈（stencil halo）置 0（就地修改并返回）。"""
    p[..., :_HALO, :] = 0
    p[..., -_HALO:, :] = 0
    p[..., :, :_HALO] = 0
    p[..., :, -_HALO:] = 0
    return p


def _rtm_wave_stencil_core(p_prev, p_curr, velocity, damp_x, damp_z,
                           num_steps, dt, dx, compute_dtype):
    """核心计算：以 compute_dtype 精度执行 num_steps 步 8 阶 stencil 时间步进。"""
    pp = _zero_halo(p_prev.to(compute_dtype).clone())      # [B, Nz, Nx]
    pc = _zero_halo(p_curr.to(compute_dtype).clone())      # [B, Nz, Nx]
    v = velocity.to(compute_dtype)
    dz = damp_z.to(compute_dtype)                          # [Nz]
    dxp = damp_x.to(compute_dtype)                         # [Nx]

    # 海绵衰减因子 g[i,j] = 1 / (1 + damp_z[i] + damp_x[j])，广播到 [Nz, Nx]
    g = 1.0 / (1.0 + dz.unsqueeze(-1) + dxp.unsqueeze(0))
    v_dt_sq = (v * dt) ** 2                                # [B, Nz, Nx]
    dx2 = dx * dx

    for _ in range(num_steps):
        # 8 阶叉形 stencil：lap = 2*c0*p + Σ_k c_k*((p[i-k]+p[i+k]) + (p[j-k]+p[j+k]))
        # halo 恒为 0，roll 的环回值全部落在 halo 内且每步后置 0，不污染内点
        lap = (2.0 * _FD8_C0) * pc
        for k, ck in enumerate(_FD8_CK, start=1):
            lap = lap + ck * (
                (torch.roll(pc, k, dims=-2) + torch.roll(pc, -k, dims=-2))
                + (torch.roll(pc, k, dims=-1) + torch.roll(pc, -k, dims=-1))
            )
        pn = (2.0 * pc - pp + (v_dt_sq * lap) / dx2) * g
        _zero_halo(pn)
        pp, pc = pc, pn
    return pp.contiguous(), pc.contiguous()


def rtm_wave_stencil(
    p_prev: torch.Tensor,
    p_curr: torch.Tensor,
    velocity: torch.Tensor,
    damp_x: torch.Tensor,
    damp_z: torch.Tensor,
    num_steps: int,
    dt: float,
    dx: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    RTM 8 阶声波 stencil 时间步进 golden reference（plain golden = bench：fp32 计算）

    Args:
        p_prev: [B, Nz, Nx] 时间片 t-1 的波场（算子先将最外 4 圈置 0）
        p_curr: [B, Nz, Nx] 时间片 t 的波场（算子先将最外 4 圈置 0）
        velocity: [B, Nz, Nx] 介质声速场（评测取值范围 [1500, 4500] m/s）
        damp_x: [Nx] 海绵层 x 方向衰减剖面（评测取值范围 [0, 0.05]，任意随机剖面合法）
        damp_z: [Nz] 海绵层 z 方向衰减剖面（评测取值范围 [0, 0.05]，任意随机剖面合法）
        num_steps: 时间步数（评测取值 1 ~ 64）
        dt: 时间步长，单位 s（评测取值范围 [2e-4, 8e-4]；case 保证 v_max*dt/dx ≤ 0.4）
        dx: 空间网格间距，单位 m（评测取值范围 [5, 20]，z/x 同间距）

    Returns:
        p_out_prev: [B, Nz, Nx] 最终时间片 t+num_steps-1，dtype 与 p_prev 一致
        p_out_curr: [B, Nz, Nx] 最终时间片 t+num_steps，dtype 与 p_prev 一致
    """
    p_out_prev, p_out_curr = _rtm_wave_stencil_core(
        p_prev, p_curr, velocity, damp_x, damp_z,
        int(num_steps), float(dt), float(dx), torch.float32)
    return p_out_prev.to(p_prev.dtype), p_out_curr.to(p_prev.dtype)


def rtm_wave_stencil_oracle(
    p_prev: torch.Tensor,
    p_curr: torch.Tensor,
    velocity: torch.Tensor,
    damp_x: torch.Tensor,
    damp_z: torch.Tensor,
    num_steps: int,
    dt: float,
    dx: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Oracle (g)：dtype-agnostic，计算精度跟随输入（fp64_cpu 下为 fp64 真值），不硬编码 .float()。"""
    return _rtm_wave_stencil_core(
        p_prev, p_curr, velocity, damp_x, damp_z,
        int(num_steps), float(dt), float(dx), p_prev.dtype)
