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
LOW_VEC_BYTES = MAX_N_ALIGN * 2

# ══ UB 地址 — 来源: DESIGN.md §3 片上地址映射表 ══
# 所有首地址均为 32B 对齐
VA_IO_LOW = 0x00000
VA_IN_FP32 = 0x10000
VA_OUT_FP32 = VA_IN_FP32 + SLOT_BYTES
VA_GAMMA_LOW = VA_OUT_FP32 + SLOT_BYTES
VA_GAMMA_FP32 = VA_GAMMA_LOW + LOW_VEC_BYTES


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

@pl.jit(auto_mutex=True, name="rms_norm_kernel_c4")
def rms_norm_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    gamma: pl.Tensor[[1, pl.DYNAMIC], pl.DT_BF16],
    eps: pl.DT_FP32,
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
):
    """RMSNorm kernel — 单 section_vector, strided 多核切分.

    Tile 布局:
      - io_low_group: bf16 输入/输出暂存（同一地址分阶段复用）
      - in_fp32_group / out_fp32_group: fp32 RMSNorm 计算
      - gamma_low_group / gamma_fp32_group: gamma 在片上升精度后复用

    所有 dtype 转换均由本 PyPTO-Pro kernel 的 ``pl.cast`` 完成；host 热路径
    不调用 Torch/ACLNN cast 或 copy 算子。
    """
    low_tile_type = pl.TileType(shape=[TILE_ROWS, MAX_N_ALIGN], dtype=pl.DT_BF16,
                                target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    fp32_tile_type = pl.TileType(shape=[TILE_ROWS, MAX_N_ALIGN], dtype=pl.DT_FP32,
                                 target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    low_vec_type = pl.TileType(shape=[1, MAX_N_ALIGN], dtype=pl.DT_BF16,
                               target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    fp32_vec_type = pl.TileType(shape=[1, MAX_N_ALIGN], dtype=pl.DT_FP32,
                                target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])

    io_low_group = pl.make_tile_group(type=low_tile_type, addrs=[VA_IO_LOW], mutex_ids=[0])
    in_fp32_group = pl.make_tile_group(type=fp32_tile_type, addrs=[VA_IN_FP32], mutex_ids=[1])
    out_fp32_group = pl.make_tile_group(type=fp32_tile_type, addrs=[VA_OUT_FP32], mutex_ids=[2])
    gamma_low_group = pl.make_tile_group(type=low_vec_type, addrs=[VA_GAMMA_LOW], mutex_ids=[3])
    gamma_fp32_group = pl.make_tile_group(type=fp32_vec_type, addrs=[VA_GAMMA_FP32], mutex_ids=[4])

    with pl.section_vector():
        rows = x.shape[0]            # N (动态)
        cols = x.shape[1]            # D (动态, 固定=2)

        # ── SPMD 原语 ──
        num_cores = pl.get_block_num()
        core_id = pl.get_block_idx()

        # ── Ceiling division ──
        num_tiles = (rows + TILE_ROWS - 1) // TILE_ROWS

        # ── 加载 gamma（每 core 一次，所有 row-tile 复用）──
        gamma_low_slot = gamma_low_group.next()
        pl.set_validshape(gamma_low_slot, [1, cols])
        pl.load(gamma_low_slot, gamma, [0, 0])
        gamma_fp32_slot = gamma_fp32_group.next()
        pl.set_validshape(gamma_fp32_slot, [1, cols])
        pl.cast(gamma_fp32_slot, gamma_low_slot, mode=pl.RoundMode.CAST_NONE)

        # ── Strided row-tile loop ──
        for tile_id in pl.range(core_id, num_tiles, num_cores):
            m_off = tile_id * TILE_ROWS
            valid_rows = pl.min(TILE_ROWS, rows - m_off)   # 尾块处理: 满 tile=TILE_ROWS, 末 tile=余数

            io_low_slot = io_low_group.next()
            pl.set_validshape(io_low_slot, [valid_rows, cols])
            pl.load(io_low_slot, x, [m_off, 0])

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
            pl.store(y, io_low_slot, [m_off, 0])

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

    host 只执行元数据 reshape、输出分配和 kernel launch；bf16↔fp32 转换
    在 kernel 内完成，避免依赖内置 Torch/ACLNN 计算 kernel。

    Args:
        x: 输入张量 shape (..., D), dtype bfloat16
        gamma: 缩放参数 shape (D,), dtype bfloat16
        epsilon: 数值稳定性参数，默认 1e-6

    Returns:
        y: RMS 归一化输出，shape 与 x 相同，dtype bfloat16
    """
    orig_shape = x.shape

    # host 适配: reshape 多维 → 2D [N, D]
    D = x.shape[-1]
    x_2d = x.reshape(-1, D)
    N = x_2d.shape[0]

    gamma_2d = gamma.reshape(1, D)
    y = torch.empty_like(x_2d)

    # 核数计算: min(物理核数, tile 数), 至少 1 核
    num_tiles = (N + TILE_ROWS - 1) // TILE_ROWS
    num_cores = min(32, max(1, num_tiles))

    # ══ 单次 kernel launch ══
    rms_norm_kernel[None, num_cores](x_2d, gamma_2d, float(epsilon), y)
    torch.npu.synchronize()

    return y.reshape(orig_shape)


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
