import torch
import tilelang
from tilelang import language as T
from ._common import PASS_CONFIGS, CAST_MODE_LOW2HIGH, CAST_MODE_HIGH2LOW, torch_dtype_to_tl

_kernel_cache = {}


@tilelang.jit(out_idx=[1], pass_configs=PASS_CONFIGS)
def _online_softmax(M, N, block_M, block_N, dtype="float16"):
    use_float32_compute = dtype in ["bfloat16", "float16"]
    cal_dtype = "float32" if use_float32_compute else dtype

    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2
    sub_block_M = block_M // VEC_NUM

    def cast_or_copy(dst, src, mode, count):
        if use_float32_compute:
            return T.tile.cast(dst, src, mode, count)
        else:
            return T.copy(src, dst)

    @T.prim_func
    def main(
        A: T.Tensor([M, N], dtype),
        B: T.Tensor([M, N], dtype),
    ):
        T.func_attr({"enable_auto_sync": True})
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
                T.copy(
                    A[bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M, by * block_N : (by + 1) * block_N],
                    a,
                    pad_value=-T.infinity(cal_dtype),
                )
                cast_or_copy(a_cal, a, CAST_MODE_LOW2HIGH, sub_block_M * block_N)
                T.reduce_max(a_cal, tile_max, dim=-1)
                T.tile.max(tile_max, prev_max, tile_max)
                T.tile.sub(tmp_exp, prev_max, tile_max)
                T.tile.exp(tmp_exp, tmp_exp)
                T.tile.mul(tmp_exp, prev_sum, tmp_exp)
                T.tile.broadcast(tile_max_2d, tile_max)
                T.tile.sub(a_cal, a_cal, tile_max_2d)
                T.tile.exp(a_cal, a_cal)
                T.reduce_sum(a_cal, tile_sum, dim=-1)
                T.tile.add(prev_sum, tile_sum, tmp_exp)
                T.copy(tile_max, prev_max)

            T.tile.broadcast(prev_max_2d, prev_max)
            T.tile.broadcast(prev_sum_2d, prev_sum)
            for by in T.serial(n_num):
                T.copy(
                    A[bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M, by * block_N : (by + 1) * block_N], a
                )
                cast_or_copy(a_cal, a, CAST_MODE_LOW2HIGH, sub_block_M * block_N)
                T.tile.sub(a_cal, a_cal, prev_max_2d)
                T.tile.exp(a_cal, a_cal)
                T.tile.div(a_cal, a_cal, prev_sum_2d)
                cast_or_copy(a, a_cal, CAST_MODE_HIGH2LOW, sub_block_M * block_N)
                T.copy(
                    a, B[bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M, by * block_N : (by + 1) * block_N]
                )

    return main


def _get_kernel(M, N, tl_dtype):
    key = (M, N, tl_dtype)
    if key not in _kernel_cache:
        block_M = 128
        block_N = 128
        _kernel_cache[key] = _online_softmax(M, N, block_M, block_N, dtype=tl_dtype)
    return _kernel_cache[key]


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    original_shape = x.shape
    ndim = x.ndim
    if ndim == 0:
        return x

    dim = dim % ndim
    original_dtype = x.dtype

    if dim != ndim - 1:
        perm = list(range(ndim))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        x = x.permute(perm).contiguous()
    else:
        x = x.contiguous()

    transposed_shape = x.shape
    last_dim = transposed_shape[-1]
    outer = 1
    for s in transposed_shape[:-1]:
        outer *= s
    x_2d = x.reshape(outer, last_dim)

    tl_dtype = torch_dtype_to_tl(original_dtype)
    kernel = _get_kernel(outer, last_dim, tl_dtype)
    out_2d = kernel(x_2d)

    out = out_2d.reshape(transposed_shape)

    if dim != ndim - 1:
        inv_perm = [0] * ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        out = out.permute(inv_perm).contiguous()

    return out
