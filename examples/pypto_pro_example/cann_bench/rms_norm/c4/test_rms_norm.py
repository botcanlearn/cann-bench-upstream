#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

"""PyPTO-Pro rms_norm kernel implementation + tests.

rms_norm: y = x / sqrt(mean(x^2) + eps) * gamma

本文件包含：
  - rms_norm_kernel: @pl.jit(auto_mutex=True) 单一 kernel
  - rms_norm_rows_vf: @pl.vector_function VF 计算函数
  - rms_norm_wrapper: 外部调用入口（host 适配 + kernel launch）
  - 4 个 test_ 函数：覆盖 DESIGN.md §8 目标测试 case

设计依据: custom/rms_norm/c4/DESIGN.md (Stage 3)
参考实现: custom/rms_norm/c1/test_rms_norm.py
权威样例: pro_ops/vf_api/test_layernorm_tile_group_vf.py
"""

import logging
import torch
import torch_npu
import pypto_pro.language as pl

logging.basicConfig(level=logging.INFO, format="%(message)s")

# ============================================================================
# 编译期常量 — 必须在 kernel 函数外部定义（模块级）
# 来源: DESIGN.md §2.1
# ============================================================================
LANES = 64                    # fp32 VF 寄存器宽度（lane 数）
MAX_N_ALIGN = LANES           # compile-time D 维度对齐值 = 64（=LANES，保证 vf.load_align 行对齐）
TILE_ROWS = 256               # 每 tile 处理的行数（沿 N 方向）
SLOT_BYTES = TILE_ROWS * MAX_N_ALIGN * 4    # fp32 tile 单 slot 字节数 = 256*64*4 = 65536
VEC_BYTES = MAX_N_ALIGN * 4                 # gamma tile [1, 64] 字节数 = 256

# ══ UB 地址 — 来源: DESIGN.md §3 片上地址映射表 ══
# 所有首地址均为 32B 对齐
VA_IN0   = 0x00000             # in_group slot 0 (ping) — 0 B
VA_IN1   = 0x10000             # in_group slot 1 (pong) — 65536 B = VA_IN0 + SLOT_BYTES
VA_OUT   = 0x20000             # out_group slot        — 131072 B = VA_IN1 + SLOT_BYTES
VA_GAMMA = 0x30000             # gamma_group slot      — 196608 B = VA_OUT + SLOT_BYTES


# ============================================================================
# VF 计算函数 — 来源: DESIGN.md §1 API 映射序列
# ============================================================================

@pl.vector_function
def rms_norm_rows_vf(
    in_tile, out_tile, gamma_tile,
    n_rows: pl.DT_INT64, n_cols: pl.DT_INT64,
    eps: pl.DT_FP32,
):
    """RMSNorm over n_cols=2 columns of each of n_rows rows.

    D=2 固定且远小于 LANES=64，每行仅 1 个 VF 寄存器。无需多寄存器循环。
    行 m 在 UB 中的元素偏移 = m * MAX_N_ALIGN (= m * 64)。

    每行算法:
        1. sum_sq = reduce_sum(x^2) → lane 0
        2. rms = sqrt(sum_sq / D + eps) → broadcast to all lanes
        3. y = x / rms * gamma（masked to valid lanes 0,1）
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    # d_reg broadcasts D=2.0 to all 64 lanes for /D division
    d_reg = vf.full(n_cols, preg, dtype=pl.DT_FP32)

    for m in pl.range(0, n_rows):
        base = m * MAX_N_ALIGN                            # 行偏移 = m * 64 (element offset)
        mreg = vf.update_mask(n_cols, dtype=pl.DT_FP32)   # n_cols=2 → mask lanes 0,1 only

        # ── Pass 1: 计算 rms = sqrt(mean(x^2) + eps) ──
        reg = vf.load_align(in_tile, base)                # 加载整行 (64 fp32 元素, 仅 lane 0,1 有效)
        sq = vf.mul(reg, reg, mreg)                       # x^2
        sum_sq = vf.reduce_sum(sq, mreg)                  # Σx^2 → lane 0
        rms_b = vf.full(sum_sq, preg)                     # broadcast lane 0 → all lanes
        rms_b = vf.div(rms_b, d_reg, preg)                # / D (÷2.0)
        rms_b = vf.adds(rms_b, eps, preg)                 # + epsilon
        rms_b = vf.sqrt(rms_b, preg)                      # sqrt (fp32 only)

        # ── Pass 2: 归一化 y = x / rms * gamma ──
        reg = vf.load_align(in_tile, base)                # 重新加载 x（in_tile 未被修改）
        gamma_reg = vf.load_align(gamma_tile, 0)          # gamma[0:64] → reg（lane 0,1 有效）
        norm = vf.div(reg, rms_b, mreg)                   # x / rms
        out_reg = vf.mul(norm, gamma_reg, mreg)            # * gamma
        vf.store_align(out_tile + base, out_reg, mreg)     # 写回 out_tile


# ============================================================================
# Kernel 函数 — 单 Phase, 纯 Vector
# 来源: DESIGN.md §4 循环与 Section 结构 + §5 分核策略 + §7 尾块处理
# ============================================================================

@pl.jit(auto_mutex=True)
def rms_norm_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    gamma: pl.Tensor[[1, pl.DYNAMIC], pl.DT_FP32],
    eps: pl.DT_FP32,
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    """RMSNorm kernel — 单 section_vector, strided 多核切分.

    Tile 布局 (DESIGN.md §3):
      - in_group:    [TILE_ROWS, MAX_N_ALIGN] fp32, 双缓冲 (VA_IN0/VA_IN1), mutex_ids=[0,1]
      - out_group:   [TILE_ROWS, MAX_N_ALIGN] fp32, 单缓冲 (VA_OUT),         mutex_ids=[2]
      - gamma_group: [1, MAX_N_ALIGN] fp32,        单缓冲 (VA_GAMMA),        mutex_ids=[3]
    """
    tile_type = pl.TileType(shape=[TILE_ROWS, MAX_N_ALIGN], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    vec_type = pl.TileType(shape=[1, MAX_N_ALIGN], dtype=pl.DT_FP32,
                           target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])

    in_group    = pl.make_tile_group(type=tile_type, addrs=[VA_IN0, VA_IN1], mutex_ids=[0, 1])
    out_group   = pl.make_tile_group(type=tile_type, addrs=[VA_OUT],         mutex_ids=[2])
    gamma_group = pl.make_tile_group(type=vec_type,  addrs=[VA_GAMMA],       mutex_ids=[3])

    with pl.section_vector():
        rows = x.shape[0]            # N (动态)
        cols = x.shape[1]            # D (动态, 固定=2)

        # ── SPMD 原语 ──
        num_cores = pl.get_block_num()
        core_id = pl.get_block_idx()

        # ── Ceiling division ──
        num_tiles = (rows + TILE_ROWS - 1) // TILE_ROWS

        # ── 加载 gamma（每 core 一次，所有 row-tile 复用）──
        gamma_slot = gamma_group.next()
        pl.set_validshape(gamma_slot, [1, cols])
        pl.load(gamma_slot, gamma, [0, 0])

        # ── Strided row-tile loop ──
        for tile_id in pl.range(core_id, num_tiles, num_cores):
            m_off = tile_id * TILE_ROWS
            valid_rows = pl.min(TILE_ROWS, rows - m_off)   # 尾块处理: 满 tile=TILE_ROWS, 末 tile=余数

            in_slot = in_group.next()
            pl.set_validshape(in_slot, [valid_rows, cols])
            pl.load(in_slot, x, [m_off, 0])                # sync: auto_mutex managed (MTE2→V)

            out_slot = out_group.next()
            pl.set_validshape(out_slot, [valid_rows, cols])

            rms_norm_rows_vf(in_slot, out_slot, gamma_slot,
                             valid_rows, cols, eps)

            pl.store(y, out_slot, [m_off, 0])              # sync: auto_mutex managed (V→MTE3)

    return


# ============================================================================
# 精度校验辅助函数
# ============================================================================

def _assert_precision(actual, *inputs, label="", **kwargs):
    """方案A精度校验（混合容差标准）。

    内部完成: CPU golden 计算 + precision_compare 对比 + PASS/FAIL 判定。
    阈值由 precision_compare 按 actual.dtype 自动查表，禁止外部覆盖。

    Args:
        actual: 算子输出 tensor (NPU, bf16)
        *inputs: 传给 golden_cpu 的位置参数 (x, gamma, 均为 NPU tensor)
        label: 测试标签（用于日志输出）
        **kwargs: 传给 golden_cpu 的关键字参数 (epsilon=...)

    Raises:
        AssertionError: 精度不达标时抛出
    """
    # ⚠️ import 必须留在函数体内，禁止提到模块顶层（交付态安全）
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
# 来源: DESIGN.md §0 维度契约（cast 策略）+ c1 参考实现
# ============================================================================

def rms_norm_wrapper(x: torch.Tensor, gamma: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """RMS Normalization 外部调用入口。

    公式: y = x / sqrt(mean(x^2) + eps) * gamma

    host 适配：
      - 多维 → 2D reshape
      - bf16 → fp32 cast（host 侧 cast，kernel 全 fp32 计算）
      - kernel launch
      - fp32 → bf16 cast 回原 dtype
      - reshape 回原 shape

    Args:
        x: 输入张量 shape (..., D), dtype bfloat16
        gamma: 缩放参数 shape (D,), dtype bfloat16
        epsilon: 数值稳定性参数，默认 1e-6

    Returns:
        y: RMS 归一化输出，shape 与 x 相同，dtype bfloat16
    """
    out_dtype = x.dtype
    orig_shape = x.shape

    # host 适配: reshape 多维 → 2D [N, D]
    D = x.shape[-1]
    x_2d = x.reshape(-1, D)
    N = x_2d.shape[0]

    # cast bf16 → fp32（kernel 内部全 fp32 计算）
    x_fp32 = x_2d.to(torch.float32)
    gamma_fp32 = gamma.to(torch.float32).reshape(1, D)

    # 输出分配
    y_fp32 = torch.empty_like(x_fp32)

    # 核数计算: min(物理核数, tile 数), 至少 1 核
    num_tiles = (N + TILE_ROWS - 1) // TILE_ROWS
    num_cores = min(32, max(1, num_tiles))

    # ══ 单次 kernel launch ══
    rms_norm_kernel[None, num_cores](x_fp32, gamma_fp32, float(epsilon), y_fp32)
    torch.npu.synchronize()

    # cast fp32 → bf16, reshape 回原 shape
    y_out = y_fp32.to(out_dtype)
    return y_out.reshape(orig_shape)


# ============================================================================
# 测试函数 — 4 个 case, 覆盖 DESIGN.md §8「目标测试 case」
# 所有 test 通过 wrapper 调 kernel，不直接调 rms_norm_kernel
# ============================================================================

def test_rms_norm_aligned():
    """case: N 全整除 TILE_ROWS (N=256，单 tile), bf16."""
    from rms_norm_golden import _get_device
    device = _get_device()
    torch.manual_seed(42)

    N, D = 256, 2
    x = torch.randn(N, D, device=device, dtype=torch.bfloat16)
    gamma = torch.randn(D, device=device, dtype=torch.bfloat16)
    y = rms_norm_wrapper(x, gamma)

    _assert_precision(y, x, gamma, epsilon=1e-6, label="aligned N=256")


def test_rms_norm_tail():
    """case: N 单轴尾块 (N=257, 1 满 tile + 尾块 1 行), bf16."""
    from rms_norm_golden import _get_device
    device = _get_device()
    torch.manual_seed(42)

    N, D = 257, 2
    x = torch.randn(N, D, device=device, dtype=torch.bfloat16)
    gamma = torch.randn(D, device=device, dtype=torch.bfloat16)
    y = rms_norm_wrapper(x, gamma)

    _assert_precision(y, x, gamma, epsilon=1e-6, label="tail N=257")


def test_rms_norm_multitile():
    """case: N 跨多 tile + 尾块 (N=513, 2 满 tile + 尾块 1 行), bf16."""
    from rms_norm_golden import _get_device
    device = _get_device()
    torch.manual_seed(42)

    N, D = 513, 2
    x = torch.randn(N, D, device=device, dtype=torch.bfloat16)
    gamma = torch.randn(D, device=device, dtype=torch.bfloat16)
    y = rms_norm_wrapper(x, gamma)

    _assert_precision(y, x, gamma, epsilon=1e-6, label="multitile N=513")


def test_rms_norm_benchmark():
    """P0 benchmark case: N=1,000,003, D=2, bf16, ~3907 tiles, 32 核 strided."""
    from rms_norm_golden import _get_device
    device = _get_device()
    torch.manual_seed(42)

    N, D = 1000003, 2
    x = torch.randn(N, D, device=device, dtype=torch.bfloat16)
    gamma = torch.randn(D, device=device, dtype=torch.bfloat16)
    y = rms_norm_wrapper(x, gamma)

    _assert_precision(y, x, gamma, epsilon=1e-6, label="benchmark N=1000003")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    logging.info("rms_norm (c4) — Pure Vec (Pattern B+ Reduce+Broadcast+Params, D=2)")
    logging.info("=" * 60)

    test_rms_norm_aligned()
    test_rms_norm_tail()
    test_rms_norm_multitile()
    test_rms_norm_benchmark()

    logging.info("=" * 60)
    logging.info("All 4 tests PASS!")
