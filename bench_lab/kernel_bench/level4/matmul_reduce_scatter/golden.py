#!/usr/bin/python3
# coding=utf-8

import torch
from typing import Optional


def matmul_reduce_scatter(
    x1: torch.Tensor,
    x2: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    hcomm_info: str = "",
    world_size: int = 1,
    reduce_op: str = "sum",
    is_trans_b: bool = False,
) -> torch.Tensor:
    """Single-process CPU reference for the MC2 matmul reduce-scatter task.

    The real benchmark path for cases with ``mc2_distributed: true`` uses
    ``MC2DistributedEvaluator`` and HCCL. This function keeps the task usable
    for metadata inspection and golden-only smoke checks.
    """
    del hcomm_info
    if reduce_op != "sum":
        raise ValueError(f"Unsupported reduce_op: {reduce_op}")
    x2_eff = x2.t() if is_trans_b else x2
    out = torch.matmul(x1.float(), x2_eff.float())
    if bias is not None:
        out = out + bias.float()
    chunk = out.shape[0] // int(world_size)
    # Single-process fallback: return rank-0 chunk
    return out[:chunk].to(x1.dtype)


def mc2_distributed_golden(ctx, inputs, attrs):
    """Distributed golden hook used by MC2DistributedEvaluator."""
    dist = ctx["dist"]
    x1 = inputs["x1"]
    x2 = inputs["x2"]
    bias = inputs.get("bias")
    x2_eff = x2.t() if bool(attrs.get("is_trans_b", False)) else x2
    out = torch.matmul(x1, x2_eff)
    if bias is not None:
        out = out + bias
    out_shape = [out.shape[0] // int(ctx["world_size"]), out.shape[1]]
    scatter = torch.empty(out_shape, dtype=out.dtype, device=out.device)
    dist._reduce_scatter_base(scatter, out, op=dist.ReduceOp.SUM)
    return scatter
