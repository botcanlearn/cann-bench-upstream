import math
import torch
import tilelang
from tilelang import language as T
from ._common import PASS_CONFIGS, torch_dtype_to_tl

_kernel_cache = {}

NUM_CORES = 48
VEC_NUM = 2
TILE_ELEMS = 16384


@tilelang.jit(out_idx=[1], pass_configs=PASS_CONFIGS)
def _exp_kernel(
    numel,
    tile_elems,
    launch_cores,
    scale,
    shift,
    log_base,
    dtype="float16",
):
    """Persistent linear kernel with affine/base transforms fused in UB."""
    need_cast = dtype == "bfloat16"
    compute_dtype = "float32" if need_cast else dtype
    block_elems = tile_elems * VEC_NUM
    num_blocks = T.ceildiv(numel, block_elems)
    num_iters = T.ceildiv(num_blocks, launch_cores)

    @T.prim_func
    def main(
        A: T.Tensor([numel], dtype),
        B: T.Tensor([numel], dtype),
    ):
        T.func_attr({"enable_auto_sync": True})
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            compute = T.alloc_ub([tile_elems], compute_dtype)
            raw_in = T.alloc_ub([tile_elems], dtype)
            raw_out = T.alloc_ub([tile_elems], dtype)

            for i in T.serial(num_iters):
                block_id = cid + i * launch_cores
                if block_id < num_blocks:
                    offset = (block_id * VEC_NUM + vid) * tile_elems
                    if offset < numel:
                        if need_cast:
                            T.copy(A[offset], raw_in)
                            T.tile.cast(compute, raw_in, "CAST_NONE", tile_elems)
                        else:
                            T.copy(A[offset], compute)

                        if scale != 1.0:
                            T.tile.mul(compute, compute, scale)
                        if shift != 0.0:
                            T.tile.add(compute, compute, shift)
                        if log_base != 1.0:
                            T.tile.mul(compute, compute, log_base)

                        T.tile.exp(compute, compute)

                        if need_cast:
                            T.tile.cast(raw_out, compute, "CAST_RINT", tile_elems)
                            T.copy(raw_out, B[offset])
                        else:
                            T.copy(compute, B[offset])

    return main


def _get_kernel(numel, tl_dtype, scale, shift, log_base):
    num_blocks = (numel + TILE_ELEMS * VEC_NUM - 1) // (TILE_ELEMS * VEC_NUM)
    launch_cores = min(NUM_CORES, num_blocks)
    key = (numel, tl_dtype, scale, shift, log_base, launch_cores)
    if key not in _kernel_cache:
        _kernel_cache[key] = _exp_kernel(
            numel,
            TILE_ELEMS,
            launch_cores,
            scale,
            shift,
            log_base,
            dtype=tl_dtype,
        )
    return _kernel_cache[key]


def exp(
    x: torch.Tensor,
    base: float = -1.0,
    scale: float = 1.0,
    shift: float = 0.0,
) -> torch.Tensor:
    original_dtype = x.dtype
    original_shape = x.shape

    x_flat = x.contiguous().reshape(-1)
    numel = x_flat.numel()
    log_base = math.log(base) if base > 0 else 1.0

    tl_dtype = torch_dtype_to_tl(original_dtype)
    kernel = _get_kernel(
        numel,
        tl_dtype,
        float(scale),
        float(shift),
        float(log_base),
    )
    return kernel(x_flat).reshape(original_shape)
