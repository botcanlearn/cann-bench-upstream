"""Softmax operator implementation using TileLang-Ascend.

Online safe softmax with 2D kernel + Python wrapper for arbitrary dim.
Supports float16, float32, bfloat16.

Algorithm (online safe softmax, two-pass):
    Pass 1: online update running max + running sum
    Pass 2: normalize output
"""

import tilelang
from tilelang import language as T
import torch

# ========== Configuration ==========
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}

pass_configs_autosync = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}

CAST_MODE_LOW2HIGH = "CAST_NONE"
CAST_MODE_HIGH2LOW = "CAST_RINT"
VEC_NUM = 2

_DTYPE_MAP = {
    torch.float16: "float16",
    torch.float32: "float",
    torch.bfloat16: "bfloat16",
}

_kernel_cache = {}


# ========== Kernel ==========
@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def online_softmax(M, N, block_M, block_N, dtype="float"):
    """Safe softmax with online normalizer (2D, dim=last).

    Supports float, float16, and bfloat16.
    fp16/bf16 use float32 compute internally.
    Non-aligned N uses pad_value=-inf for tail blocks.
    """
    use_float32_compute = dtype in ["bfloat16", "float16"]
    cal_dtype = "float32" if use_float32_compute else dtype

    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    sub_block_M = block_M // VEC_NUM

    use_db = use_float32_compute and (block_M <= 32) and (n_num >= 2)

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            bx = cid
            a = T.alloc_ub([sub_block_M, block_N], dtype)
            a_cal = T.alloc_ub([sub_block_M, block_N], cal_dtype)
            tile_max = T.alloc_ub([sub_block_M, 1], cal_dtype)
            tile_max_2d = T.alloc_ub([sub_block_M, block_N], cal_dtype)
            prev_max = T.alloc_ub([sub_block_M, 1], cal_dtype)
            prev_max_2d = T.alloc_ub([sub_block_M, block_N], cal_dtype)
            tile_sum = T.alloc_ub([sub_block_M, 1], cal_dtype)
            prev_sum = T.alloc_ub([sub_block_M, 1], cal_dtype)
            prev_sum_2d = T.alloc_ub([sub_block_M, block_N], cal_dtype)
            tmp_exp = T.alloc_ub([sub_block_M, 1], cal_dtype)

            T.tile.fill(prev_max, -T.infinity(cal_dtype))
            T.tile.fill(prev_sum, 0.0)

            # Pass 1: online update running max + running sum
            for by in T.serial(n_num):
                if use_float32_compute:
                    T.copy(
                        A[
                            bx * block_M + vid * sub_block_M : bx * block_M
                            + (vid + 1) * sub_block_M,
                            by * block_N : (by + 1) * block_N,
                        ],
                        a,
                        pad_value=-T.infinity(cal_dtype),
                    )
                    T.barrier_all()
                    T.tile.cast(a_cal, a, CAST_MODE_LOW2HIGH, sub_block_M * block_N)
                else:
                    T.copy(
                        A[
                            bx * block_M + vid * sub_block_M : bx * block_M
                            + (vid + 1) * sub_block_M,
                            by * block_N : (by + 1) * block_N,
                        ],
                        a_cal,
                        pad_value=-T.infinity(cal_dtype),
                    )
                    T.barrier_all()
                T.reduce_max(a_cal, tile_max, dim=-1)
                T.tile.max(tile_max, prev_max, tile_max)
                T.tile.sub(tmp_exp, prev_max, tile_max)
                T.tile.exp(tmp_exp, tmp_exp)

                T.tile.broadcast(tile_max_2d, tile_max)
                T.tile.sub(a_cal, a_cal, tile_max_2d)
                T.tile.exp(a_cal, a_cal)
                T.reduce_sum(a_cal, tile_sum, dim=-1)
                T.tile.mul_add_dst(tile_sum, tmp_exp, prev_sum)
                T.copy(tile_sum, prev_sum)
                T.copy(tile_max, prev_max)
                T.barrier_all()

            # Pass 2: normalize output
            T.tile.broadcast(prev_max_2d, prev_max)
            T.tile.broadcast(prev_sum_2d, prev_sum)
            T.barrier_all()
            if use_db:
                a_p2 = T.alloc_ub([2, sub_block_M, block_N], dtype)
                a_cal_p2 = T.alloc_ub([sub_block_M, block_N], cal_dtype)
                out_p2 = T.alloc_ub([2, sub_block_M, block_N], dtype)

                T.set_flag("mte3", "mte2", 0)
                T.set_flag("mte3", "mte2", 1)

                T.wait_flag("mte3", "mte2", 0)
                T.copy(
                    A[
                        bx * block_M + vid * sub_block_M : bx * block_M
                        + (vid + 1) * sub_block_M,
                        0 : block_N,
                    ],
                    a_p2[0, :, :],
                    pad_value=-T.infinity(cal_dtype),
                )
                T.set_flag("mte2", "v", 0)

                for by in T.serial(n_num):
                    cur = by % 2
                    nxt = (by + 1) % 2
                    if by < n_num - 1:
                        T.wait_flag("mte3", "mte2", nxt)
                        T.copy(
                            A[
                                bx * block_M + vid * sub_block_M : bx * block_M
                                + (vid + 1) * sub_block_M,
                                (by + 1) * block_N : (by + 2) * block_N,
                            ],
                            a_p2[nxt, :, :],
                            pad_value=-T.infinity(cal_dtype),
                        )
                        T.set_flag("mte2", "v", nxt)
                    T.wait_flag("mte2", "v", cur)
                    T.tile.cast(
                        a_cal_p2, a_p2[cur, :, :], CAST_MODE_LOW2HIGH,
                        sub_block_M * block_N,
                    )
                    T.tile.sub(a_cal_p2, a_cal_p2, prev_max_2d)
                    T.tile.exp(a_cal_p2, a_cal_p2)
                    T.tile.div(a_cal_p2, a_cal_p2, prev_sum_2d)
                    T.tile.cast(
                        out_p2[cur, :, :], a_cal_p2, CAST_MODE_HIGH2LOW,
                        sub_block_M * block_N,
                    )
                    T.set_flag("v", "mte3", cur)
                    T.wait_flag("v", "mte3", cur)
                    T.copy(
                        out_p2[cur, :, :],
                        B[
                            bx * block_M + vid * sub_block_M : bx * block_M
                            + (vid + 1) * sub_block_M,
                            by * block_N : (by + 1) * block_N,
                        ],
                    )
                    T.set_flag("mte3", "mte2", cur)

                T.wait_flag("mte3", "mte2", 0)
                T.wait_flag("mte3", "mte2", 1)
            else:
                for by in T.serial(n_num):
                    if use_float32_compute:
                        T.copy(
                            A[
                                bx * block_M + vid * sub_block_M : bx * block_M
                                + (vid + 1) * sub_block_M,
                                by * block_N : (by + 1) * block_N,
                            ],
                            a,
                            pad_value=-T.infinity(cal_dtype),
                        )
                        T.barrier_all()
                        T.tile.cast(a_cal, a, CAST_MODE_LOW2HIGH, sub_block_M * block_N)
                    else:
                        T.copy(
                            A[
                                bx * block_M + vid * sub_block_M : bx * block_M
                                + (vid + 1) * sub_block_M,
                                by * block_N : (by + 1) * block_N,
                            ],
                            a_cal,
                            pad_value=-T.infinity(cal_dtype),
                        )
                        T.barrier_all()
                    T.tile.sub(a_cal, a_cal, prev_max_2d)
                    T.tile.exp(a_cal, a_cal)
                    T.tile.div(a_cal, a_cal, prev_sum_2d)
                    if use_float32_compute:
                        T.tile.cast(a, a_cal, CAST_MODE_HIGH2LOW, sub_block_M * block_N)
                        T.barrier_all()
                        T.copy(
                            a,
                            B[
                                bx * block_M + vid * sub_block_M : bx * block_M
                                + (vid + 1) * sub_block_M,
                                by * block_N : (by + 1) * block_N,
                            ],
                        )
                    else:
                        T.barrier_all()
                        T.copy(
                            a_cal,
                            B[
                                bx * block_M + vid * sub_block_M : bx * block_M
                                + (vid + 1) * sub_block_M,
                                by * block_N : (by + 1) * block_N,
                            ],
                        )
                    T.barrier_all()

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs_autosync)
def online_softmax_autosync(M, N, block_M, block_N, dtype="float"):
    """Safe softmax with online normalizer (2D, dim=last) — AUTO_SYNC=True variant."""
    use_float32_compute = dtype in ["bfloat16", "float16"]
    cal_dtype = "float32" if use_float32_compute else dtype

    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    sub_block_M = block_M // VEC_NUM

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            bx = cid
            a = T.alloc_ub([sub_block_M, block_N], dtype)
            a_cal = T.alloc_ub([sub_block_M, block_N], cal_dtype)
            tile_max = T.alloc_ub([sub_block_M, 1], cal_dtype)
            tile_max_2d = T.alloc_ub([sub_block_M, block_N], cal_dtype)
            prev_max = T.alloc_ub([sub_block_M, 1], cal_dtype)
            prev_max_2d = T.alloc_ub([sub_block_M, block_N], cal_dtype)
            tile_sum = T.alloc_ub([sub_block_M, 1], cal_dtype)
            prev_sum = T.alloc_ub([sub_block_M, 1], cal_dtype)
            prev_sum_2d = T.alloc_ub([sub_block_M, block_N], cal_dtype)
            tmp_exp = T.alloc_ub([sub_block_M, 1], cal_dtype)

            T.tile.fill(prev_max, -T.infinity(cal_dtype))
            T.tile.fill(prev_sum, 0.0)

            for by in T.serial(n_num):
                if use_float32_compute:
                    T.copy(
                        A[
                            bx * block_M + vid * sub_block_M : bx * block_M
                            + (vid + 1) * sub_block_M,
                            by * block_N : (by + 1) * block_N,
                        ],
                        a,
                        pad_value=-T.infinity(cal_dtype),
                    )
                    T.tile.cast(a_cal, a, CAST_MODE_LOW2HIGH, sub_block_M * block_N)
                else:
                    T.copy(
                        A[
                            bx * block_M + vid * sub_block_M : bx * block_M
                            + (vid + 1) * sub_block_M,
                            by * block_N : (by + 1) * block_N,
                        ],
                        a_cal,
                        pad_value=-T.infinity(cal_dtype),
                    )
                T.reduce_max(a_cal, tile_max, dim=-1)
                T.tile.max(tile_max, prev_max, tile_max)
                T.tile.sub(tmp_exp, prev_max, tile_max)
                T.tile.exp(tmp_exp, tmp_exp)

                T.tile.broadcast(tile_max_2d, tile_max)
                T.tile.sub(a_cal, a_cal, tile_max_2d)
                T.tile.exp(a_cal, a_cal)
                T.reduce_sum(a_cal, tile_sum, dim=-1)
                T.tile.mul_add_dst(tile_sum, tmp_exp, prev_sum)
                T.copy(tile_sum, prev_sum)
                T.copy(tile_max, prev_max)

            T.tile.broadcast(prev_max_2d, prev_max)
            T.tile.broadcast(prev_sum_2d, prev_sum)
            for by in T.serial(n_num):
                if use_float32_compute:
                    T.copy(
                        A[
                            bx * block_M + vid * sub_block_M : bx * block_M
                            + (vid + 1) * sub_block_M,
                            by * block_N : (by + 1) * block_N,
                        ],
                        a,
                        pad_value=-T.infinity(cal_dtype),
                    )
                    T.tile.cast(a_cal, a, CAST_MODE_LOW2HIGH, sub_block_M * block_N)
                else:
                    T.copy(
                        A[
                            bx * block_M + vid * sub_block_M : bx * block_M
                            + (vid + 1) * sub_block_M,
                            by * block_N : (by + 1) * block_N,
                        ],
                        a_cal,
                        pad_value=-T.infinity(cal_dtype),
                    )
                T.tile.sub(a_cal, a_cal, prev_max_2d)
                T.tile.exp(a_cal, a_cal)
                T.tile.div(a_cal, a_cal, prev_sum_2d)
                if use_float32_compute:
                    T.tile.cast(a, a_cal, CAST_MODE_HIGH2LOW, sub_block_M * block_N)
                    T.copy(
                        a,
                        B[
                            bx * block_M + vid * sub_block_M : bx * block_M
                            + (vid + 1) * sub_block_M,
                            by * block_N : (by + 1) * block_N,
                        ],
                    )
                else:
                    T.copy(
                        a_cal,
                        B[
                            bx * block_M + vid * sub_block_M : bx * block_M
                            + (vid + 1) * sub_block_M,
                            by * block_N : (by + 1) * block_N,
                        ],
                    )

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs_autosync)
def online_softmax_dim0(M, N, block_M, block_N, dtype="float"):
    """Safe softmax with online normalizer (2D, dim=0 — reduce along M)."""
    use_float32_compute = dtype in ["bfloat16", "float16"]
    cal_dtype = "float32" if use_float32_compute else dtype

    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    sub_block_N = block_N // VEC_NUM

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(n_num, is_npu=True) as (cid, vid):
            bx = cid
            col_start = bx * block_N + vid * sub_block_N
            col_end = bx * block_N + (vid + 1) * sub_block_N

            a = T.alloc_ub([block_M, sub_block_N], dtype)
            a_cal = T.alloc_ub([block_M, sub_block_N], cal_dtype)
            tile_max = T.alloc_ub([1, sub_block_N], cal_dtype)
            tile_max_2d = T.alloc_ub([block_M, sub_block_N], cal_dtype)
            prev_max = T.alloc_ub([1, sub_block_N], cal_dtype)
            prev_max_2d = T.alloc_ub([block_M, sub_block_N], cal_dtype)
            tile_sum = T.alloc_ub([1, sub_block_N], cal_dtype)
            prev_sum = T.alloc_ub([1, sub_block_N], cal_dtype)
            prev_sum_2d = T.alloc_ub([block_M, sub_block_N], cal_dtype)
            tmp_exp = T.alloc_ub([1, sub_block_N], cal_dtype)

            T.tile.fill(prev_max, -T.infinity(cal_dtype))
            T.tile.fill(prev_sum, 0.0)

            for bx_m in T.serial(m_num):
                row_start = bx_m * block_M
                if use_float32_compute:
                    T.copy(
                        A[row_start : row_start + block_M, col_start : col_end],
                        a,
                        pad_value=-T.infinity(cal_dtype),
                    )
                    T.tile.cast(a_cal, a, CAST_MODE_LOW2HIGH, block_M * sub_block_N)
                else:
                    T.copy(
                        A[row_start : row_start + block_M, col_start : col_end],
                        a_cal,
                        pad_value=-T.infinity(cal_dtype),
                    )
                T.reduce_max(a_cal, tile_max, dim=0)
                T.tile.max(tile_max, prev_max, tile_max)
                T.tile.sub(tmp_exp, prev_max, tile_max)
                T.tile.exp(tmp_exp, tmp_exp)

                T.tile.broadcast(tile_max_2d, tile_max, axis=0)
                T.tile.sub(a_cal, a_cal, tile_max_2d)
                T.tile.exp(a_cal, a_cal)
                T.reduce_sum(a_cal, tile_sum, dim=0)
                T.tile.mul_add_dst(tile_sum, tmp_exp, prev_sum)
                T.copy(tile_sum, prev_sum)
                T.copy(tile_max, prev_max)

            T.tile.broadcast(prev_max_2d, prev_max, axis=0)
            T.tile.broadcast(prev_sum_2d, prev_sum, axis=0)
            for bx_m in T.serial(m_num):
                row_start = bx_m * block_M
                if use_float32_compute:
                    T.copy(
                        A[row_start : row_start + block_M, col_start : col_end],
                        a,
                        pad_value=-T.infinity(cal_dtype),
                    )
                    T.tile.cast(a_cal, a, CAST_MODE_LOW2HIGH, block_M * sub_block_N)
                else:
                    T.copy(
                        A[row_start : row_start + block_M, col_start : col_end],
                        a_cal,
                        pad_value=-T.infinity(cal_dtype),
                    )
                T.tile.sub(a_cal, a_cal, prev_max_2d)
                T.tile.exp(a_cal, a_cal)
                T.tile.div(a_cal, a_cal, prev_sum_2d)
                if use_float32_compute:
                    T.tile.cast(a, a_cal, CAST_MODE_HIGH2LOW, block_M * sub_block_N)
                    T.copy(a, B[row_start : row_start + block_M, col_start : col_end])
                else:
                    T.copy(a_cal, B[row_start : row_start + block_M, col_start : col_end])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs_autosync)
def online_softmax_3d(A_dim, B_dim, C_dim, block_B, block_C, dtype="float"):
    """Safe softmax along dim=1 of 3D [A, B, C] tensor — no permute needed."""
    use_float32_compute = dtype in ["bfloat16", "float16"]
    cal_dtype = "float32" if use_float32_compute else dtype

    b_num = T.ceildiv(B_dim, block_B)
    c_num = T.ceildiv(C_dim, block_C)
    m_num = A_dim * c_num
    sub_block_C = block_C // VEC_NUM

    @T.prim_func
    def main(
        X: T.Tensor((A_dim, B_dim, C_dim), dtype),  # type: ignore
        Y: T.Tensor((A_dim, B_dim, C_dim), dtype),  # type: ignore
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            a_idx = cid // c_num
            c_blk = cid % c_num
            c_start = c_blk * block_C + vid * sub_block_C
            c_end = c_start + sub_block_C

            a_ub = T.alloc_ub([block_B, sub_block_C], dtype)
            a_cal = T.alloc_ub([block_B, sub_block_C], cal_dtype)
            tile_max = T.alloc_ub([1, sub_block_C], cal_dtype)
            tile_max_2d = T.alloc_ub([block_B, sub_block_C], cal_dtype)
            prev_max = T.alloc_ub([1, sub_block_C], cal_dtype)
            prev_max_2d = T.alloc_ub([block_B, sub_block_C], cal_dtype)
            tile_sum = T.alloc_ub([1, sub_block_C], cal_dtype)
            prev_sum = T.alloc_ub([1, sub_block_C], cal_dtype)
            prev_sum_2d = T.alloc_ub([block_B, sub_block_C], cal_dtype)
            tmp_exp = T.alloc_ub([1, sub_block_C], cal_dtype)

            T.tile.fill(prev_max, -T.infinity(cal_dtype))
            T.tile.fill(prev_sum, 0.0)

            for bx_b in T.serial(b_num):
                b_start = bx_b * block_B
                if use_float32_compute:
                    T.copy(
                        X[a_idx, b_start : b_start + block_B, c_start : c_end],
                        a_ub,
                        pad_value=-T.infinity(cal_dtype),
                    )
                    T.tile.cast(a_cal, a_ub, CAST_MODE_LOW2HIGH, block_B * sub_block_C)
                else:
                    T.copy(
                        X[a_idx, b_start : b_start + block_B, c_start : c_end],
                        a_cal,
                        pad_value=-T.infinity(cal_dtype),
                    )
                T.reduce_max(a_cal, tile_max, dim=0)
                T.tile.max(tile_max, prev_max, tile_max)
                T.tile.sub(tmp_exp, prev_max, tile_max)
                T.tile.exp(tmp_exp, tmp_exp)

                T.tile.broadcast(tile_max_2d, tile_max, axis=0)
                T.tile.sub(a_cal, a_cal, tile_max_2d)
                T.tile.exp(a_cal, a_cal)
                T.reduce_sum(a_cal, tile_sum, dim=0)
                T.tile.mul_add_dst(tile_sum, tmp_exp, prev_sum)
                T.copy(tile_sum, prev_sum)
                T.copy(tile_max, prev_max)

            T.tile.broadcast(prev_max_2d, prev_max, axis=0)
            T.tile.broadcast(prev_sum_2d, prev_sum, axis=0)
            for bx_b in T.serial(b_num):
                b_start = bx_b * block_B
                if use_float32_compute:
                    T.copy(
                        X[a_idx, b_start : b_start + block_B, c_start : c_end],
                        a_ub,
                        pad_value=-T.infinity(cal_dtype),
                    )
                    T.tile.cast(a_cal, a_ub, CAST_MODE_LOW2HIGH, block_B * sub_block_C)
                else:
                    T.copy(
                        X[a_idx, b_start : b_start + block_B, c_start : c_end],
                        a_cal,
                        pad_value=-T.infinity(cal_dtype),
                    )
                T.tile.sub(a_cal, a_cal, prev_max_2d)
                T.tile.exp(a_cal, a_cal)
                T.tile.div(a_cal, a_cal, prev_sum_2d)
                if use_float32_compute:
                    T.tile.cast(a_ub, a_cal, CAST_MODE_HIGH2LOW, block_B * sub_block_C)
                    T.copy(a_ub, Y[a_idx, b_start : b_start + block_B, c_start : c_end])
                else:
                    T.copy(a_cal, Y[a_idx, b_start : b_start + block_B, c_start : c_end])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs_autosync)
def online_softmax_single(M, N, block_M, block_N, dtype="float"):
    """Simplified softmax for n_num=1 (N <= block_N, i.e. small N)."""
    use_float32_compute = dtype in ["bfloat16", "float16"]
    cal_dtype = "float32" if use_float32_compute else dtype

    m_num = T.ceildiv(M, block_M)
    sub_block_M = block_M // VEC_NUM

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            bx = cid
            a = T.alloc_ub([sub_block_M, block_N], dtype)
            a_cal = T.alloc_ub([sub_block_M, block_N], cal_dtype)
            tile_max = T.alloc_ub([sub_block_M, 1], cal_dtype)
            tile_max_2d = T.alloc_ub([sub_block_M, block_N], cal_dtype)
            tile_sum = T.alloc_ub([sub_block_M, 1], cal_dtype)
            tile_sum_2d = T.alloc_ub([sub_block_M, block_N], cal_dtype)

            if use_float32_compute:
                T.copy(
                    A[
                        bx * block_M + vid * sub_block_M : bx * block_M
                        + (vid + 1) * sub_block_M,
                        0 : block_N,
                    ],
                    a,
                    pad_value=-T.infinity(cal_dtype),
                )
                T.tile.cast(a_cal, a, CAST_MODE_LOW2HIGH, sub_block_M * block_N)
            else:
                T.copy(
                    A[
                        bx * block_M + vid * sub_block_M : bx * block_M
                        + (vid + 1) * sub_block_M,
                        0 : block_N,
                    ],
                    a_cal,
                    pad_value=-T.infinity(cal_dtype),
                )
            T.reduce_max(a_cal, tile_max, dim=-1)
            T.tile.broadcast(tile_max_2d, tile_max)
            T.tile.sub(a_cal, a_cal, tile_max_2d)
            T.tile.exp(a_cal, a_cal)
            T.reduce_sum(a_cal, tile_sum, dim=-1)
            T.tile.broadcast(tile_sum_2d, tile_sum)
            T.tile.div(a_cal, a_cal, tile_sum_2d)
            if use_float32_compute:
                T.tile.cast(a, a_cal, CAST_MODE_HIGH2LOW, sub_block_M * block_N)
                T.copy(
                    a,
                    B[
                        bx * block_M + vid * sub_block_M : bx * block_M
                        + (vid + 1) * sub_block_M,
                        0 : block_N,
                    ],
                )
            else:
                T.copy(
                    a_cal,
                    B[
                        bx * block_M + vid * sub_block_M : bx * block_M
                        + (vid + 1) * sub_block_M,
                        0 : block_N,
                    ],
                )

    return main


def _select_block(M, N, dtype_str):
    """Select block_M, block_N based on shape and dtype."""
    core_num = 24
    if N < 128:
        block_N = 32
    else:
        block_N = 128
    n_num = (N + block_N - 1) // block_N

    # Small-N path: n_num==1 uses simplified kernel with fewer buffers → larger block_M
    if n_num == 1:
        block_M = 1024
        cal_bytes = 4
        dtype_bytes = 2 if dtype_str in ("float16", "bfloat16") else 4
        sub_bm = block_M // VEC_NUM
        # 6 buffers: 1 dtype 2D + 3 fp32 2D + 2 fp32 1D
        ub_est = sub_bm * block_N * dtype_bytes + 3 * sub_bm * block_N * cal_bytes + 2 * sub_bm * cal_bytes
        while ub_est > 185 * 1024 and block_M > 32:
            block_M //= 2
            sub_bm = block_M // VEC_NUM
            ub_est = sub_bm * block_N * dtype_bytes + 3 * sub_bm * block_N * cal_bytes + 2 * sub_bm * cal_bytes
        return block_M, block_N

    if M >= core_num * 128:
        block_M = 128
    elif n_num >= 8:
        block_M = 32
    else:
        block_M = 128

    m_num = (M + block_M - 1) // block_M
    if m_num < core_num and block_M > 16:
        for bm in [16]:
            if (M + bm - 1) // bm >= core_num:
                block_M = bm
                break

    if block_M <= 32 and N >= 256:
        sub_bm = block_M // VEC_NUM
        max_bn = 512
        # UB guard: fp16/bf16 uses 4 fp32 2D + 1 dtype 2D; fp32 uses 5 fp32 2D
        if dtype_str in ("float16", "bfloat16"):
            while max_bn > 128 and (sub_bm * max_bn * 4 * 4 + sub_bm * max_bn * 2) > 170 * 1024:
                max_bn //= 2
        else:
            while max_bn > 128 and (sub_bm * max_bn * 4 * 5 + sub_bm * 4 * 5) > 170 * 1024:
                max_bn //= 2
        bn = max_bn
        while bn > N:
            bn //= 2
        block_N = max(128, bn)
    return block_M, block_N


def _get_kernel(M, N, block_M, block_N, dtype_str):
    """Get or compile kernel for given config (with caching)."""
    use_float32_compute = dtype_str in ["bfloat16", "float16"]
    n_num = (N + block_N - 1) // block_N
    use_db = use_float32_compute and (block_M <= 32) and (n_num >= 2)

    key = (M, N, block_M, block_N, dtype_str)
    if key not in _kernel_cache:
        if n_num == 1:
            _kernel_cache[key] = online_softmax_single(
                M, N, block_M, block_N, dtype=dtype_str
            )
        elif use_db:
            _kernel_cache[key] = online_softmax(M, N, block_M, block_N, dtype=dtype_str)
        else:
            _kernel_cache[key] = online_softmax_autosync(
                M, N, block_M, block_N, dtype=dtype_str
            )
    return _kernel_cache[key]


def _select_block_dim0(M, N, dtype_str):
    """Select block sizes for dim=0 kernel (reduce along M, parallel across N)."""
    core_num = 24
    block_M = 128
    block_N = 128

    n_num = (N + block_N - 1) // block_N
    if n_num < core_num and block_N > 32:
        block_N = max(32, (N + core_num - 1) // core_num)
        block_N = ((block_N + 15) // 16) * 16

    sub_bn = block_N // VEC_NUM
    cal_bytes = 4 if dtype_str in ("float16", "bfloat16") else 4
    ub_est = 5 * block_M * sub_bn * cal_bytes
    while ub_est > 170 * 1024 and block_M > 16:
        block_M //= 2
        ub_est = 5 * block_M * sub_bn * cal_bytes

    return block_M, block_N


_kernel_cache_dim0 = {}


def _get_kernel_dim0(M, N, block_M, block_N, dtype_str):
    """Get or compile dim=0 kernel for given config (with caching)."""
    key = (M, N, block_M, block_N, dtype_str)
    if key not in _kernel_cache_dim0:
        _kernel_cache_dim0[key] = online_softmax_dim0(
            M, N, block_M, block_N, dtype=dtype_str
        )
    return _kernel_cache_dim0[key]


def _select_block_3d(A_dim, B_dim, C_dim, dtype_str):
    """Select block sizes for 3D interior-dim kernel (reduce along B, parallel across A*C)."""
    block_B = 128
    block_C = 128

    c_num = (C_dim + block_C - 1) // block_C
    m_num = A_dim * c_num
    core_num = 24

    if m_num < core_num and block_C > 32:
        block_C = max(32, (C_dim * A_dim + core_num - 1) // core_num)
        block_C = ((block_C + 15) // 16) * 16
        if block_C < 16:
            block_C = 16

    sub_bc = block_C // VEC_NUM
    cal_bytes = 4
    ub_est = 10 * block_B * sub_bc * cal_bytes
    while ub_est > 170 * 1024 and block_B > 16:
        block_B //= 2
        ub_est = 10 * block_B * sub_bc * cal_bytes

    return block_B, block_C


_kernel_cache_3d = {}


def _get_kernel_3d(A_dim, B_dim, C_dim, block_B, block_C, dtype_str):
    """Get or compile 3D interior-dim kernel for given config (with caching)."""
    key = (A_dim, B_dim, C_dim, block_B, block_C, dtype_str)
    if key not in _kernel_cache_3d:
        _kernel_cache_3d[key] = online_softmax_3d(
            A_dim, B_dim, C_dim, block_B, block_C, dtype=dtype_str
        )
    return _kernel_cache_3d[key]


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax implementation: Python wrapper + TileLang kernel.

    Args:
        x: input tensor (1~8D), dtype in {float16, float32, bfloat16}
        dim: softmax dimension, supports negative index

    Returns:
        output tensor with same shape and dtype as x
    """
    rank = x.dim()
    if dim < 0:
        dim = dim + rank
    assert 0 <= dim < rank, f"dim {dim} out of range for rank {rank}"

    dtype_str = _DTYPE_MAP[x.dtype]

    # Fast path 1: dim == last and rank == 2
    if dim == rank - 1 and rank == 2:
        M, N = x.shape
        block_M, block_N = _select_block(M, N, dtype_str)
        kernel = _get_kernel(M, N, block_M, block_N, dtype_str)
        return kernel(x)

    # Fast path 2: dim == last (any rank) — just flatten to 2D
    if dim == rank - 1 and x.is_contiguous():
        N = x.shape[-1]
        M = x.numel() // N
        x_2d = x.reshape(M, N)
        block_M, block_N = _select_block(M, N, dtype_str)
        kernel = _get_kernel(M, N, block_M, block_N, dtype_str)
        y_2d = kernel(x_2d)
        return y_2d.reshape(x.shape)

    # Fast path 3: dim == 0 and rank == 2
    if dim == 0 and rank == 2 and x.is_contiguous():
        M, N = x.shape
        block_M, block_N = _select_block_dim0(M, N, dtype_str)
        kernel = _get_kernel_dim0(M, N, block_M, block_N, dtype_str)
        return kernel(x)

    # Fast path 4: interior dim (not 0, not last)
    if dim != 0 and dim != rank - 1 and x.is_contiguous():
        outer = 1
        for i in range(dim):
            outer *= x.shape[i]
        B_dim = x.shape[dim]
        inner = 1
        for i in range(dim + 1, rank):
            inner *= x.shape[i]
        x_3d = x.reshape(outer, B_dim, inner)
        block_B, block_C = _select_block_3d(outer, B_dim, inner, dtype_str)
        kernel = _get_kernel_3d(outer, B_dim, inner, block_B, block_C, dtype_str)
        y_3d = kernel(x_3d)
        return y_3d.reshape(x.shape)

    # General path: permute dim to last, flatten to 2D
    perm = [i for i in range(rank) if i != dim] + [dim]
    x_perm = x.permute(perm).contiguous()
    N = x.shape[dim]
    M = x_perm.numel() // N
    x_2d = x_perm.reshape(M, N)

    block_M, block_N = _select_block(M, N, dtype_str)
    kernel = _get_kernel(M, N, block_M, block_N, dtype_str)
    y_2d = kernel(x_2d)

    y_perm = y_2d.reshape(x_perm.shape)
    inv_perm = [0] * rank
    for i, p in enumerate(perm):
        inv_perm[p] = i
    y = y_perm.permute(inv_perm).contiguous()
    return y
