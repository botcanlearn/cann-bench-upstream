"""
CANN Bench Utils - Framework warmup and cache clean operators

Provides two custom operators for v3 anti-cheat:
- cann_bench_warmup: MatMul (10240x10240, fp16) for NPU frequency boost
- cann_bench_cache_clean: ReduceMax (96x1024x1024, fp16) for L2 cache flush

These operators use specialized naming (CannBenchWarmup/CannBenchCacheClean)
for profiling filtering without shape matching.
"""

import torch
import torch_npu  # noqa: F401  ensure NPU backend (PrivateUse1) is initialized

from . import _C

torch.library.define("cann_bench_utils::cann_bench_warmup", "(Tensor x, Tensor y) -> Tensor")
torch.library.define("cann_bench_utils::cann_bench_cache_clean", "(Tensor x) -> Tensor")


@torch.library.impl("cann_bench_utils::cann_bench_warmup", "PrivateUse1")
def _warmup_npu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    z = torch.empty_like(x)
    _C.warmup_npu(x, y, z)
    return z


@torch.library.impl("cann_bench_utils::cann_bench_warmup", "Meta")
def _warmup_meta(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _C.warmup_meta(x, y)


@torch.library.impl("cann_bench_utils::cann_bench_cache_clean", "PrivateUse1")
def _cache_clean_npu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty((), dtype=x.dtype, device=x.device)
    _C.cache_clean_npu(x, out)
    return out


@torch.library.impl("cann_bench_utils::cann_bench_cache_clean", "Meta")
def _cache_clean_meta(x: torch.Tensor) -> torch.Tensor:
    return _C.cache_clean_meta(x)


def cann_bench_warmup(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """NPU warmup operation (MatMul for frequency boost).

    Args:
        x: Input tensor (10240, 10240), dtype=float16, device='npu'
        y: Input tensor (10240, 10240), dtype=float16, device='npu'

    Returns:
        Output tensor (10240, 10240), dtype=float16

    Note:
        This is NOT a full MatMul implementation - just enough to boost NPU frequency.
        Profiling Type: "CannBenchWarmup"
    """
    return torch.ops.cann_bench_utils.cann_bench_warmup(x, y)


def cann_bench_cache_clean(x: torch.Tensor) -> torch.Tensor:
    """L2 cache clean operation (ReduceMax for cache flush).

    Args:
        x: Input tensor (96, 1024, 1024), dtype=float16, device='npu'

    Returns:
        Scalar tensor, dtype=float16

    Note:
        This is NOT a full ReduceMax implementation - just enough to flush L2 cache.
        Profiling Type: "CannBenchCacheClean"
    """
    return torch.ops.cann_bench_utils.cann_bench_cache_clean(x)


__all__ = ['cann_bench_warmup', 'cann_bench_cache_clean']
