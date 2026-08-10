#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

"""PyPTO-Pro rms_norm kernel implementation + tests.

rms_norm: y = x / sqrt(mean(x^2) + eps) * gamma

本文件包含：
  - rms_norm_kernel: @pl.jit(auto_mutex=True) 单一 kernel
  - rms_norm_rows_vf: @pl.vector_function VF 计算函数
  - rms_norm_wrapper:  外部调用入口（host 适配 + kernel launch）
  - 5 个 test_ 函数：覆盖 DESIGN.md §8 用户指定 case (c2)

设计依据: custom/rms_norm/c2/DESIGN.md (Stage 3)
参考样例: pro_ops/vf_api/test_layernorm_tile_group_vf.py
"""

import logging
import os
import torch
import torch_npu
import pypto_pro.language as pl

logging.basicConfig(level=logging.INFO, format="%(message)s")

# ============================================================================
# 编译期常量 — 必须在 kernel 函数外部定义（模块级）
# 来源: DESIGN.md §2.1
# ============================================================================
LANES = 64               # fp32 VF 寄存器宽度（元素数）
MAX_N = 8192             # 编译期 tile 宽度（LANES 倍数: 8192=64*128）
TILE_ROWS = 1            # 每 tile 处理的行数
SLOT_BYTES = TILE_ROWS * MAX_N * 4    # 1 * 8192 * 4 = 32768 B
VEC_BYTES = MAX_N * 4                  # 8192 * 4 = 32768 B

# UB 地址 — 来源: DESIGN.md §3
VA_IN0 = 0x00000           #       0 B  — in_group slot 0
VA_IN1 = 0x08000           #   32768 B  — in_group slot 1
VA_OUT0 = 0x10000          #   65536 B  — out_group slot 0
VA_OUT1 = 0x18000          #   98304 B  — out_group slot 1
VA_GAMMA = 0x20000         #  131072 B  — gamma slot


# ============================================================================
# VF 计算函数
# ============================================================================

@pl.vector_function
def rms_norm_rows_vf(
    in_tile, out_tile, gamma_tile,
    n_rows: pl.DT_INT64, n_cols: pl.DT_INT64,
    eps: pl.DT_FP32,
):
    """RMSNorm over n_cols columns of each of n_rows rows.

    Each row spans ``n_regs = ceil(n_cols / LANES)`` VF registers.
    Row ``m`` starts at UB element offset ``m * MAX_N``.
    gamma register ``r`` at element offset ``r * LANES``.

    Algorithm (per row):
        1. sum_sq = sum(x^2)  across all n_regs registers
        2. rms = sqrt(sum_sq / D + eps)
        3. y = x / rms * gamma  across all registers
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    n_regs = (n_cols + LANES - 1) // LANES
    # n_reg_f broadcasts D (as fp32 scalar) to all lanes for /D divide
    n_reg_f = vf.full(n_cols, preg, dtype=pl.DT_FP32)

    for m in pl.range(0, n_rows):
        base = m * MAX_N

        # ---- Pass 1: sum_sq = Σx² across all registers ----
        sum_sq = vf.full(0.0, preg, dtype=pl.DT_FP32)
        for r in pl.range(0, n_regs):
            valid = pl.min(LANES, n_cols - r * LANES)
            mreg = vf.update_mask(valid, dtype=pl.DT_FP32)
            reg = vf.load_align(in_tile, base + r * LANES)
            sq = vf.mul(reg, reg, mreg)                  # x²
            part = vf.reduce_sum(sq, mreg)               # per-reg sum → lane0
            sum_sq = vf.add(sum_sq, part, preg)          # accumulate into lane0

        # ---- Compute rms = sqrt(Σx² / D + eps) ----
        mean_sq_b = vf.full(sum_sq, preg)                # broadcast lane0 → all lanes
        mean_sq_b = vf.div(mean_sq_b, n_reg_f, preg)     # / D
        mean_sq_b = vf.adds(mean_sq_b, eps, preg)        # + eps
        rms_b = vf.sqrt(mean_sq_b, preg)                 # sqrt → rms broadcast

        # ---- Pass 2: y = x / rms * gamma ----
        for r in pl.range(0, n_regs):
            valid = pl.min(LANES, n_cols - r * LANES)
            mreg = vf.update_mask(valid, dtype=pl.DT_FP32)
            reg = vf.load_align(in_tile, base + r * LANES)
            gamma_reg = vf.load_align(gamma_tile, r * LANES)
            norm = vf.div(reg, rms_b, mreg)              # x / rms
            out = vf.mul(norm, gamma_reg, mreg)           # * gamma
            vf.store_align(out_tile + (base + r * LANES), out, mreg)


# ============================================================================
# Kernel 函数 (单 Phase, 纯 Vector)
# ============================================================================

@pl.jit(auto_mutex=True, name="rms_norm_kernel_c2")
def rms_norm_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    gamma: pl.Tensor[[1, pl.DYNAMIC], pl.DT_FP32],
    eps: pl.DT_FP32,
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    """RMSNorm kernel — 单 section_vector, strided 多核切分.

    Tile 布局 (DESIGN.md §3):
      - in_group:  [TILE_ROWS, MAX_N] fp32, 双缓冲 (VA_IN0/VA_IN1), mutex_ids=[0,1]
      - out_group: [TILE_ROWS, MAX_N] fp32, 双缓冲 (VA_OUT0/VA_OUT1), mutex_ids=[2,3]
      - gamma_group: [1, MAX_N] fp32,    单缓冲 (VA_GAMMA),      mutex_ids=[4]
    """
    tile_type = pl.TileType(shape=[TILE_ROWS, MAX_N], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    vec_type = pl.TileType(shape=[1, MAX_N], dtype=pl.DT_FP32,
                           target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])

    in_group = pl.make_tile_group(type=tile_type, addrs=[VA_IN0, VA_IN1], mutex_ids=[0, 1])
    out_group = pl.make_tile_group(type=tile_type, addrs=[VA_OUT0, VA_OUT1], mutex_ids=[2, 3])
    gamma_group = pl.make_tile_group(type=vec_type, addrs=[VA_GAMMA], mutex_ids=[4])

    with pl.section_vector():
        rows = x.shape[0]
        cols = x.shape[1]
        num_cores = pl.get_block_num()
        core_id = pl.get_block_idx()

        # gamma 每 core 加载一次，所有 row-tile 复用
        gamma_slot = gamma_group.next()
        pl.set_validshape(gamma_slot, [1, cols])
        pl.load(gamma_slot, gamma, [0, 0])

        # Ceiling division for row-tile count
        num_tiles = (rows + TILE_ROWS - 1) // TILE_ROWS

        for tile_id in pl.range(core_id, num_tiles, num_cores):
            row_off = tile_id * TILE_ROWS
            valid_rows = pl.min(TILE_ROWS, rows - row_off)

            in_slot = in_group.next()
            pl.set_validshape(in_slot, [valid_rows, cols])
            pl.load(in_slot, x, [row_off, 0])

            out_slot = out_group.next()
            pl.set_validshape(out_slot, [valid_rows, cols])
            rms_norm_rows_vf(in_slot, out_slot, gamma_slot, valid_rows, cols, eps)

            pl.store(y, out_slot, [row_off, 0])

    return


# ============================================================================
# 精度校验辅助函数
# ============================================================================

def _assert_precision(actual, *inputs, label="", **kwargs):
    """方案A精度校验（混合容差标准）。

    Args:
        actual: kernel 输出 tensor (NPU fp32)
        *inputs: 传给 golden_cpu 的位置参数（x, gamma）
        label: 测试标签
        **kwargs: 传给 golden_cpu 的关键字参数（epsilon=...）
    """
    # dev-only import, 不放在模块顶层避免交付态 ModuleNotFoundError
    from precision_compare import check_precision
    from rms_norm_golden_cpu import rms_norm_golden_cpu

    inputs_cpu = [i.cpu() if hasattr(i, "cpu") else i for i in inputs]
    golden = rms_norm_golden_cpu(*inputs_cpu, **kwargs)
    actual_cpu = actual.cpu() if hasattr(actual, "cpu") else actual
    passed, summary = check_precision(actual_cpu, golden)
    if not passed:
        raise AssertionError(f"精度不达标 [{label}]: {summary}")
    if label:
        logging.info("[%s] PASS (%s)", label, summary)
    return summary


# ============================================================================
# 入口函数 — 外部调用入口
# ============================================================================

def rms_norm_wrapper(x: torch.Tensor, gamma: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """RMS Normalization 外部调用入口。

    公式: y = x / sqrt(mean(x^2) + eps) * gamma

    Args:
        x: 输入张量 shape (..., D), dtype float32
        gamma: 缩放参数 shape (D,), dtype float32
        epsilon: 数值稳定性参数，默认 1e-6

    Returns:
        y: RMS 归一化输出，shape 与 x 相同，dtype float32
    """
    orig_shape = x.shape

    # host 适配: reshape 多维 → 2D [N, D]
    D = x.shape[-1]
    x_2d = x.reshape(-1, D)
    N = x_2d.shape[0]

    # kernel 内部全 fp32 计算（输入已是 fp32）
    x_fp32 = x_2d.to(torch.float32)
    gamma_fp32 = gamma.to(torch.float32).reshape(1, D)

    # 输出分配
    y_fp32 = torch.empty_like(x_fp32)

    # 核数计算
    num_tiles = (N + TILE_ROWS - 1) // TILE_ROWS
    num_cores = min(32, max(1, num_tiles))

    # 单次 kernel launch
    rms_norm_kernel[None, num_cores](x_fp32, gamma_fp32, float(epsilon), y_fp32)
    torch.npu.synchronize()

    return y_fp32.reshape(orig_shape)


# ============================================================================
# 测试函数 — 5 个 benchmark cases (DESIGN.md §8 用户指定 case)
# ============================================================================

def _make_inputs(device, x_shape, gamma_dim, eps, value_range):
    """构造测试输入（与 cases.yaml / DESIGN.md §8 一致）。"""
    lo, hi = value_range
    x = torch.empty(x_shape, dtype=torch.float32, device=device).uniform_(lo, hi)
    # gamma 用 ones 确保输出一致性（gamma=1 时输出 RMS ≈ 1）
    gamma = torch.ones(gamma_dim, dtype=torch.float32, device=device)
    return x, gamma, eps


def test_case_2():
    """case_2: [32,128,1024]+[1024], eps=1e-6, range=[-2,2]. 大 rows(4096), 对齐 D=1024(16 regs)."""
    from rms_norm_golden import _get_device
    device = _get_device()
    torch.manual_seed(42)
    x, gamma, eps = _make_inputs(device, [32, 128, 1024], 1024, 1e-6, (-2.0, 2.0))
    y = rms_norm_wrapper(x, gamma, epsilon=eps)
    _assert_precision(y, x, gamma, epsilon=eps, label="case_2")


def test_case_5():
    """case_5: [8,512,8192]+[8192], eps=1e-6, range=[-100,100]. 最大 D=8192=MAX_N(128 regs)."""
    from rms_norm_golden import _get_device
    device = _get_device()
    torch.manual_seed(42)
    x, gamma, eps = _make_inputs(device, [8, 512, 8192], 8192, 1e-6, (-100.0, 100.0))
    y = rms_norm_wrapper(x, gamma, epsilon=eps)
    _assert_precision(y, x, gamma, epsilon=eps, label="case_5")


def test_case_8():
    """case_8: [16,511,2049]+[2049], eps=1e-4, range=[-1,1]. 非对齐 D=2049(33 regs=32full+1tail)."""
    from rms_norm_golden import _get_device
    device = _get_device()
    torch.manual_seed(42)
    x, gamma, eps = _make_inputs(device, [16, 511, 2049], 2049, 1e-4, (-1.0, 1.0))
    y = rms_norm_wrapper(x, gamma, epsilon=eps)
    _assert_precision(y, x, gamma, epsilon=eps, label="case_8")


def test_case_11():
    """case_11: [31,129,2049]+[2049], eps=1e-6, range=[-50,100]. 非对齐 D=2049, 奇数 rows=3999."""
    from rms_norm_golden import _get_device
    device = _get_device()
    torch.manual_seed(42)
    x, gamma, eps = _make_inputs(device, [31, 129, 2049], 2049, 1e-6, (-50.0, 100.0))
    y = rms_norm_wrapper(x, gamma, epsilon=eps)
    _assert_precision(y, x, gamma, epsilon=eps, label="case_11")


def test_case_14():
    """case_14: [11,367,373]+[373], eps=1e-5, range=[-10,10]. 小 D=373(6 regs=5full+1tail=53 lanes)."""
    from rms_norm_golden import _get_device
    device = _get_device()
    torch.manual_seed(42)
    x, gamma, eps = _make_inputs(device, [11, 367, 373], 373, 1e-5, (-10.0, 10.0))
    y = rms_norm_wrapper(x, gamma, epsilon=eps)
    _assert_precision(y, x, gamma, epsilon=eps, label="case_14")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    logging.info("rms_norm — Pure Vec (Pattern B+ Reduce+Broadcast+Params)")
    logging.info("=" * 60)

    test_case_2()
    test_case_5()
    test_case_8()
    test_case_11()
    test_case_14()

    logging.info("=" * 60)
    logging.info("All 5 tests PASS!")
