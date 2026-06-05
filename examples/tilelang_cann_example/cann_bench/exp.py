import math
import torch
import tilelang
from tilelang import language as T
from ._common import PASS_CONFIGS, torch_dtype_to_tl

_kernel_cache = {}


@tilelang.jit(out_idx=[1], pass_configs=PASS_CONFIGS)
def _exp_kernel(M, N, block_M, block_N, dtype="float16"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2
    sub_block_M = block_M // VEC_NUM

    @T.prim_func
    def main(
        A: T.Tensor([M, N], dtype),
        B: T.Tensor([M, N], dtype),
    ):
        T.func_attr({"enable_auto_sync": True})
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a = T.alloc_ub([sub_block_M, block_N], dtype)
            b = T.alloc_ub([sub_block_M, block_N], dtype)

            row_start = bx * block_M + vid * sub_block_M
            col_start = by * block_N

            T.copy(
                A[row_start : row_start + sub_block_M, col_start : col_start + block_N],
                a,
            )
            T.tile.exp(b, a)
            T.copy(
                b,
                B[row_start : row_start + sub_block_M, col_start : col_start + block_N],
            )

    return main


def _get_kernel(M, N, tl_dtype):
    key = (M, N, tl_dtype)
    if key not in _kernel_cache:
        block_M = 128
        block_N = 128
        _kernel_cache[key] = _exp_kernel(M, N, block_M, block_N, dtype=tl_dtype)
    return _kernel_cache[key]


def exp(
    x: torch.Tensor,
    base: float = -1.0,
    scale: float = 1.0,
    shift: float = 0.0,
) -> torch.Tensor:
    original_dtype = x.dtype
    original_shape = x.shape

    kernel_dtype = original_dtype
    if original_dtype == torch.bfloat16:
        kernel_dtype = torch.float32
        x = x.to(torch.float32)

    temp = scale * x + shift
    if base > 0:
        temp = temp * math.log(base)
    temp = temp.contiguous()

    temp_flat = temp.reshape(-1, temp.size(-1))
    M, N = temp_flat.shape

    tl_dtype = torch_dtype_to_tl(kernel_dtype)
    kernel = _get_kernel(M, N, tl_dtype)
    out_flat = kernel(temp_flat)

    out = out_flat.reshape(original_shape)
    if original_dtype == torch.bfloat16:
        out = out.to(torch.bfloat16)
    return out
