#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

"""PyPTO-Pro rms_norm kernel implementation + tests.

rms_norm: y = x / sqrt(mean(x^2) + eps) * gamma

本文件包含：
  - rms_norm_kernel: @pl.jit(auto_mutex=True) 单一 kernel
  - rms_norm_rows_vf: @pl.vector_function VF 计算函数
  - rms_norm_wrapper:  外部调用入口（host 适配 + kernel launch）
  - 6 个 test_ 函数：覆盖 DESIGN.md §8 目标测试 case

设计依据: custom/rms_norm/c1/DESIGN.md (Stage 3)
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
MAX_N = 4096              # 编译期 tile 宽度（32B 对齐，LANES 倍数: 4096=64*64）
TILE_ROWS = 4             # 每 tile 处理的行数
SLOT_BYTES = TILE_ROWS * MAX_N * 4    # fp32 tile: 4 * 4096 * 4 = 65536 B
LOW_VEC_BYTES = MAX_N * 2              # fp16 gamma: 4096 * 2 = 8192 B

# UB 地址 — 来源: DESIGN.md §3
VA_IO_LOW = 0x00000        #       0 B  — fp16 input/output staging tile
VA_IN_FP32 = 0x10000       #   65536 B  — fp32 compute input tile
VA_OUT_FP32 = VA_IN_FP32 + SLOT_BYTES
VA_GAMMA_LOW = VA_OUT_FP32 + SLOT_BYTES
VA_GAMMA_FP32 = VA_GAMMA_LOW + LOW_VEC_BYTES


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

@pl.jit(auto_mutex=True, name="rms_norm_kernel_c1")
def rms_norm_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    gamma: pl.Tensor[[1, pl.DYNAMIC], pl.DT_FP16],
    eps: pl.DT_FP32,
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    """RMSNorm kernel — 单 section_vector, strided 多核切分.

    Tile 布局:
      - io_low_group: fp16 输入/输出暂存（同一地址分阶段复用）
      - in_fp32_group / out_fp32_group: fp32 RMSNorm 计算
      - gamma_low_group / gamma_fp32_group: gamma 在片上升精度后复用

    所有 dtype 转换均由本 PyPTO-Pro kernel 的 ``pl.cast`` 完成；host 热路径
    不调用 Torch/ACLNN cast 或 copy 算子。
    """
    low_tile_type = pl.TileType(shape=[TILE_ROWS, MAX_N], dtype=pl.DT_FP16,
                                target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    fp32_tile_type = pl.TileType(shape=[TILE_ROWS, MAX_N], dtype=pl.DT_FP32,
                                 target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    low_vec_type = pl.TileType(shape=[1, MAX_N], dtype=pl.DT_FP16,
                               target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    fp32_vec_type = pl.TileType(shape=[1, MAX_N], dtype=pl.DT_FP32,
                                target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])

    io_low_group = pl.make_tile_group(type=low_tile_type, addrs=[VA_IO_LOW], mutex_ids=[0])
    in_fp32_group = pl.make_tile_group(type=fp32_tile_type, addrs=[VA_IN_FP32], mutex_ids=[1])
    out_fp32_group = pl.make_tile_group(type=fp32_tile_type, addrs=[VA_OUT_FP32], mutex_ids=[2])
    gamma_low_group = pl.make_tile_group(type=low_vec_type, addrs=[VA_GAMMA_LOW], mutex_ids=[3])
    gamma_fp32_group = pl.make_tile_group(type=fp32_vec_type, addrs=[VA_GAMMA_FP32], mutex_ids=[4])

    with pl.section_vector():
        rows = x.shape[0]
        cols = x.shape[1]
        num_cores = pl.get_block_num()
        core_id = pl.get_block_idx()

        # gamma 每 core 加载一次，所有 row-tile 复用
        gamma_low_slot = gamma_low_group.next()
        pl.set_validshape(gamma_low_slot, [1, cols])
        pl.load(gamma_low_slot, gamma, [0, 0])
        gamma_fp32_slot = gamma_fp32_group.next()
        pl.set_validshape(gamma_fp32_slot, [1, cols])
        pl.cast(gamma_fp32_slot, gamma_low_slot, mode=pl.RoundMode.CAST_NONE)

        # Ceiling division for row-tile count
        num_tiles = (rows + TILE_ROWS - 1) // TILE_ROWS

        for tile_id in pl.range(core_id, num_tiles, num_cores):
            row_off = tile_id * TILE_ROWS
            valid_rows = pl.min(TILE_ROWS, rows - row_off)

            io_low_slot = io_low_group.next()
            pl.set_validshape(io_low_slot, [valid_rows, cols])
            pl.load(io_low_slot, x, [row_off, 0])

            in_fp32_slot = in_fp32_group.next()
            pl.set_validshape(in_fp32_slot, [valid_rows, cols])
            pl.cast(in_fp32_slot, io_low_slot, mode=pl.RoundMode.CAST_NONE)

            out_fp32_slot = out_fp32_group.next()
            pl.set_validshape(out_fp32_slot, [valid_rows, cols])
            rms_norm_rows_vf(
                in_fp32_slot, out_fp32_slot, gamma_fp32_slot,
                valid_rows, cols, eps,
            )

            pl.cast(io_low_slot, out_fp32_slot, mode=pl.RoundMode.CAST_ROUND)
            pl.store(y, io_low_slot, [row_off, 0])

    return


# ============================================================================
# 精度校验辅助函数
# ============================================================================

def _assert_precision(actual, *inputs, label="", **kwargs):
    """方案A精度校验（混合容差标准）。

    Args:
        actual: kernel 输出 tensor (NPU fp16)
        *inputs: 传给 golden_cpu 的位置参数（x, gamma）
        label: 测试标签
        **kwargs: 传给 golden_cpu 的关键字参数（epsilon=...）
    """
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
        x: 输入张量 shape (..., D), dtype float16
        gamma: 缩放参数 shape (D,), dtype float16
        epsilon: 数值稳定性参数，默认 1e-6

    Returns:
        y: RMS 归一化输出，shape 与 x 相同，dtype float16
    """
    orig_shape = x.shape

    # host 适配: reshape 多维 → 2D [N, D]
    D = x.shape[-1]
    x_2d = x.reshape(-1, D)
    N = x_2d.shape[0]

    gamma_2d = gamma.reshape(1, D)
    y = torch.empty_like(x_2d)

    # 核数计算
    num_tiles = (N + TILE_ROWS - 1) // TILE_ROWS
    num_cores = min(32, max(1, num_tiles))

    # 单次 kernel launch
    rms_norm_kernel[None, num_cores](x_2d, gamma_2d, float(epsilon), y)
    torch.npu.synchronize()

    return y.reshape(orig_shape)


# ============================================================================
# Benchmark case 配置 — 来源: SPEC.md §12 + DESIGN.md §8
# ============================================================================

_BENCHMARK_CASES = [
    # (case_name, x_shape, gamma_shape, epsilon, value_range)
    ("case_1",  [32, 128, 768],    768,   1e-6,   (-1.0, 1.0)),
    ("case_4",  [16, 256, 4096],   4096,  1e-6,   (-10.0, 10.0)),
    ("case_7",  [63, 67, 1023],    1023,  1e-8,   (-0.1, 0.1)),
    ("case_10", [33, 127, 769],    769,   1e-6,   (-1.0, 2.0)),
    ("case_13", [7, 1009, 1021],   1021,  1e-7,   (-1.0, 1.0)),
    ("case_19", [4, 255, 4096],    4096,  0.001,  (-65504.0, 65504.0)),
]


# ============================================================================
# 测试函数 — 6 个 benchmark cases
# ============================================================================

def _make_inputs(device, x_shape, gamma_dim, eps, value_range):
    """构造测试输入（与 golden 输入分布一致）。"""
    lo, hi = value_range
    x = torch.empty(x_shape, dtype=torch.float32, device=device).uniform_(lo, hi).to(torch.float16)
    gamma = torch.ones(gamma_dim, dtype=torch.float16, device=device)
    return x, gamma, eps


def test_case_1():
    """case_1: 全整除 (行 4096%4=0, 列 768%64=0), 小 D, 对齐 hidden=768."""
    from rms_norm_golden import _get_device
    device = _get_device()
    torch.manual_seed(42)
    x, gamma, eps = _make_inputs(device, [32, 128, 768], 768, 1e-6, (-1.0, 1.0))
    y = rms_norm_wrapper(x, gamma, epsilon=eps)
    _assert_precision(y, x, gamma, epsilon=eps, label="case_1")


def test_case_4():
    """case_4: 全整除, 满 D=4096=MAX_N, 最大 tile 利用."""
    from rms_norm_golden import _get_device
    device = _get_device()
    torch.manual_seed(42)
    x, gamma, eps = _make_inputs(device, [16, 256, 4096], 4096, 1e-6, (-10.0, 10.0))
    y = rms_norm_wrapper(x, gamma, epsilon=eps)
    _assert_precision(y, x, gamma, epsilon=eps, label="case_4")


def test_case_7():
    """case_7: 行尾块 (4221%4=1) + 列尾 lane (1023%64=63) + 极小 eps=1e-8."""
    from rms_norm_golden import _get_device
    device = _get_device()
    torch.manual_seed(42)
    x, gamma, eps = _make_inputs(device, [63, 67, 1023], 1023, 1e-8, (-0.1, 0.1))
    y = rms_norm_wrapper(x, gamma, epsilon=eps)
    _assert_precision(y, x, gamma, epsilon=eps, label="case_7")


def test_case_10():
    """case_10: 行尾块 (4191%4=3) + 列尾 lane (769%64=1), 非对齐 hidden=769."""
    from rms_norm_golden import _get_device
    device = _get_device()
    torch.manual_seed(42)
    x, gamma, eps = _make_inputs(device, [33, 127, 769], 769, 1e-6, (-1.0, 2.0))
    y = rms_norm_wrapper(x, gamma, epsilon=eps)
    _assert_precision(y, x, gamma, epsilon=eps, label="case_10")


def test_case_13():
    """case_13: 行尾块 (7063%4=3) + 列尾 lane (1021%64=61) + 素数 hidden + 多 tile."""
    from rms_norm_golden import _get_device
    device = _get_device()
    torch.manual_seed(42)
    x, gamma, eps = _make_inputs(device, [7, 1009, 1021], 1021, 1e-7, (-1.0, 1.0))
    y = rms_norm_wrapper(x, gamma, epsilon=eps)
    _assert_precision(y, x, gamma, epsilon=eps, label="case_13")


def test_case_19():
    """case_19: float16 边界值 ±65504, 大 eps=0.001, 满 D=4096."""
    from rms_norm_golden import _get_device
    device = _get_device()
    torch.manual_seed(42)
    x, gamma, eps = _make_inputs(device, [4, 255, 4096], 4096, 0.001, (-65504.0, 65504.0))
    y = rms_norm_wrapper(x, gamma, epsilon=eps)
    _assert_precision(y, x, gamma, epsilon=eps, label="case_19")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    logging.info("rms_norm — Pure Vec (Pattern B+ Reduce+Broadcast+Params)")
    logging.info("=" * 60)

    test_case_1()
    test_case_4()
    test_case_7()
    test_case_10()
    test_case_13()
    test_case_19()

    logging.info("=" * 60)
    logging.info("All 6 tests PASS!")
