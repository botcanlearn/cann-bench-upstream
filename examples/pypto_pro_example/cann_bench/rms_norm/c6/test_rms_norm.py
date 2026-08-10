#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

"""PyPTO-Pro rms_norm kernel implementation + tests for c6 case.

rms_norm: y = x / sqrt(mean(x^2) + eps) * gamma

本文件包含：
  - rms_norm_kernel: @pl.jit(auto_mutex=True) 单一 kernel
  - rms_norm_rows_vf: @pl.vector_function VF 计算函数
  - rms_norm_wrapper:  外部调用入口（host 适配 + kernel launch）
  - 5 个 test_ 函数：覆盖 DESIGN.md §8 目标测试 case

设计依据: custom/rms_norm/c6/DESIGN.md (Stage 3)
参考实现: custom/rms_norm/c1/test_rms_norm.py（已验证同模式）
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
MAX_N = 4160              # 编译期 tile 宽度（LANES 倍数: 4160 = 65 × 64）
                          # 偏差: DESIGN.md §2.1 原定 4096，但 §8 test_03/04 需要 D=4100，
                          # 需 ceil(4100/64)=65 个寄存器 → MAX_N ≥ 4160
TILE_ROWS = 4             # 每 tile 处理的行数
SLOT_BYTES = TILE_ROWS * MAX_N * 4    # 4 × 4160 × 4 = 66560 B = 0x10400
VEC_BYTES = MAX_N * 4                  # 4160 × 4 = 16640 B = 0x4100

# UB 地址 — 来源: DESIGN.md §3（MAX_N=4160 重新计算）
VA_IN0 = 0x00000           #       0 B  — in_group slot 0 (ping)
VA_IN1 = 0x10400           #   66560 B  — in_group slot 1 (pong)
VA_OUT0 = 0x20800          #  133120 B  — out_group slot (单缓冲)
VA_GAMMA = 0x30C00         #  199680 B  — gamma slot
# UB 总用量: 3×66560 + 16640 = 216320 B / 248 KB = 85.2% (安全)


# ============================================================================
# VF 计算函数
# 来源: DESIGN.md §1 API 序列 + c1 已验证实现
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
# 来源: DESIGN.md §4 伪代码骨架 + c1 已验证实现
# ============================================================================

@pl.jit(auto_mutex=True, name="rms_norm_kernel_c6")
def rms_norm_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    gamma: pl.Tensor[[1, pl.DYNAMIC], pl.DT_FP32],
    eps: pl.DT_FP32,
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    """RMSNorm kernel — 单 section_vector, strided 多核切分.

    Tile 布局:
      - in_group:  [TILE_ROWS=4, MAX_N=4160] fp32, 双缓冲 (VA_IN0/VA_IN1), mutex_ids=[0,1]
      - out_group: [TILE_ROWS=4, MAX_N=4160] fp32, 单缓冲 (VA_OUT0),     mutex_ids=[2]
      - gamma_group: [1, MAX_N=4160] fp32,      单缓冲 (VA_GAMMA),    mutex_ids=[3]
      偏差: DESIGN.md §2.1 原定 MAX_N=4096，增至 4160 以支持 §8 test_03/04 (D=4100)
    """
    tile_type = pl.TileType(shape=[TILE_ROWS, MAX_N], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    vec_type = pl.TileType(shape=[1, MAX_N], dtype=pl.DT_FP32,
                           target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])

    in_group = pl.make_tile_group(type=tile_type, addrs=[VA_IN0, VA_IN1], mutex_ids=[0, 1])
    out_group = pl.make_tile_group(type=tile_type, addrs=[VA_OUT0], mutex_ids=[2])
    gamma_group = pl.make_tile_group(type=vec_type, addrs=[VA_GAMMA], mutex_ids=[3])

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
    # ⚠️ 这两个 import 必须留在函数体内——交付单元仅含 test_rms_norm.py +
    # rms_norm_golden.py，不含 precision_compare.py / rms_norm_golden_cpu.py
    from precision_compare import check_precision
    from rms_norm_golden_cpu import rms_norm_golden_cpu

    # 输入传给 CPU golden 前转到 CPU
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

def rms_norm_wrapper(
    x: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float = 1e-06,
) -> torch.Tensor:
    """RMS Normalization 外部调用入口。

    公式: y = x / sqrt(mean(x^2) + eps) * gamma

    Args:
        x: 输入张量 shape (..., D), dtype float16/float32/bfloat16
        gamma: 缩放参数 shape (D,), 同 dtype
        epsilon: 数值稳定性参数，默认 1e-6

    Returns:
        y: RMS 归一化输出，shape 与 x 相同，dtype 与 x 相同
    """
    out_dtype = x.dtype
    orig_shape = x.shape

    # host 适配: reshape 多维 → 2D [S, D]
    D = x.shape[-1]
    x_2d = x.reshape(-1, D)

    # cast → fp32（kernel 内部全 fp32 计算）
    x_fp32 = x_2d.to(torch.float32)
    gamma_fp32 = gamma.to(torch.float32).reshape(1, D)

    # 输出分配
    y_fp32 = torch.empty_like(x_fp32)

    # 核数计算
    num_tiles = (x_fp32.shape[0] + TILE_ROWS - 1) // TILE_ROWS
    num_cores = min(32, max(1, num_tiles))

    # 单次 kernel launch
    rms_norm_kernel[None, num_cores](x_fp32, gamma_fp32, float(epsilon), y_fp32)
    torch.npu.synchronize()

    # cast 回原始 dtype, reshape 回原 shape
    y_out = y_fp32.to(out_dtype)
    return y_out.reshape(orig_shape)


# ============================================================================
# 测试函数 — 5 个 case，来自 DESIGN.md §8「目标测试 case」
# ============================================================================

def test_01_full_divide():
    """test_01: S 整除 TILE_ROWS (8%4=0), D 整除 LANES (4096%64=0), 无尾块."""
    from rms_norm_golden import _get_device
    device = _get_device()
    torch.manual_seed(42)
    # shape: [8, 4096], fp32
    x = torch.randn(8, 4096, dtype=torch.float32, device=device)
    gamma = torch.randn(4096, dtype=torch.float32, device=device)
    y = rms_norm_wrapper(x, gamma, epsilon=1e-6)
    _assert_precision(y, x, gamma, epsilon=1e-6, label="test_01_full_divide")


def test_02_row_tail():
    """test_02: S 尾块 (7%4=3), D 全整除, 测试行维度尾块处理."""
    from rms_norm_golden import _get_device
    device = _get_device()
    torch.manual_seed(42)
    # shape: [7, 4096], fp32
    x = torch.randn(7, 4096, dtype=torch.float32, device=device)
    gamma = torch.randn(4096, dtype=torch.float32, device=device)
    y = rms_norm_wrapper(x, gamma, epsilon=1e-6)
    _assert_precision(y, x, gamma, epsilon=1e-6, label="test_02_row_tail")


def test_03_col_tail():
    """test_03: S 全整除, D 尾 lane (4100%64=4), 测试列维度尾寄存器处理."""
    from rms_norm_golden import _get_device
    device = _get_device()
    torch.manual_seed(42)
    # shape: [8, 4100], fp32
    x = torch.randn(8, 4100, dtype=torch.float32, device=device)
    gamma = torch.randn(4100, dtype=torch.float32, device=device)
    y = rms_norm_wrapper(x, gamma, epsilon=1e-6)
    _assert_precision(y, x, gamma, epsilon=1e-6, label="test_03_col_tail")


def test_04_multi_tail():
    """test_04: S 尾块 (11%4=3) + D 尾 lane (4100%64=4), 跨多 tile (3 tiles)."""
    from rms_norm_golden import _get_device
    device = _get_device()
    torch.manual_seed(42)
    # shape: [11, 4100], fp32
    x = torch.randn(11, 4100, dtype=torch.float32, device=device)
    gamma = torch.randn(4100, dtype=torch.float32, device=device)
    y = rms_norm_wrapper(x, gamma, epsilon=1e-6)
    _assert_precision(y, x, gamma, epsilon=1e-6, label="test_04_multi_tail")


def test_05_user_zero():
    """test_05: 用户指定全零输入边界测试.

    x: [3, 7, 11, 4096] 全零, fp32, eps=1e-4
    gamma: [4096] 随机非零
    预期: y 全零（x=0 → x²=0 → rms=sqrt(eps) → 0/rms=0）
    """
    from rms_norm_golden import _get_device
    device = _get_device()
    torch.manual_seed(42)
    # host: 4D [3, 7, 11, 4096] → kernel 2D [231, 4096]
    x = torch.zeros(3, 7, 11, 4096, dtype=torch.float32, device=device)
    gamma = torch.randn(4096, dtype=torch.float32, device=device)
    y = rms_norm_wrapper(x, gamma, epsilon=1e-4)
    _assert_precision(y, x, gamma, epsilon=1e-4, label="test_05_user_zero")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    logging.info("rms_norm c6 — Pure Vec (Pattern B+ Reduce+Broadcast+Params)")
    logging.info("=" * 60)

    test_01_full_divide()
    test_02_row_tail()
    test_03_col_tail()
    test_04_multi_tail()
    test_05_user_zero()

    logging.info("=" * 60)
    logging.info("All 5 tests PASS!")
