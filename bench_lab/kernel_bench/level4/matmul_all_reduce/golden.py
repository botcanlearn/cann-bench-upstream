#!/usr/bin/python3
# coding=utf-8

from typing import Optional

import torch


def _dtype_from_name(dtype_name: str):
    dtype = str(dtype_name).lower()
    if dtype in ("fp16", "float16"):
        return torch.float16
    if dtype in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype in ("fp32", "float32"):
        return torch.float32
    if dtype in ("int32",):
        return torch.int32
    if dtype in ("int64",):
        return torch.int64
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _make_tensor(shape, dtype_name: str, value_range, seed: int, device):
    if shape is None:
        return None
    torch.manual_seed(int(seed))
    dtype = _dtype_from_name(dtype_name)
    if value_range is None:
        value_range = [0, 1]
    lo, hi = value_range
    if dtype.is_floating_point:
        tensor = torch.empty(shape, dtype=torch.float32).uniform_(float(lo), float(hi)).to(dtype)
    else:
        tensor = torch.randint(int(lo), int(hi) + 1, shape, dtype=dtype)
    return tensor.to(device)


def mc2_make_rank_inputs(ctx, case_payload):
    """Build rank-local inputs with mc2_test MatmulAllReduce seed semantics."""
    shapes = case_payload["input_shapes"]
    dtypes = case_payload["dtypes"]
    ranges = case_payload.get("value_ranges") or []
    attrs = case_payload["attrs"]
    rank = int(ctx["rank"])
    device = ctx["device"]
    seed = int(attrs.get("seed", 1))

    x1 = _make_tensor(shapes[0], dtypes[0], ranges[0] if ranges else None, seed, device)
    x2 = _make_tensor(shapes[1], dtypes[1], ranges[1] if len(ranges) > 1 else None, seed, device)
    bias = None
    if bool(attrs.get("is_bias", False)) and len(shapes) > 2 and shapes[2] is not None:
        bias = _make_tensor(shapes[2], dtypes[2], ranges[2] if len(ranges) > 2 else None, seed * 7, device)
    return {"x1": x1, "x2": x2, "bias": bias}


def matmul_all_reduce(
    x1: torch.Tensor,
    x2: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    hcomm_info: str = "",
    world_size: int = 1,
    reduce_op: str = "sum",
    is_trans_b: bool = False,
) -> torch.Tensor:
    """Single-process CPU reference for the MC2 matmul all-reduce task."""
    del hcomm_info
    if reduce_op != "sum":
        raise ValueError(f"Unsupported reduce_op: {reduce_op}")
    x2_eff = x2.t() if is_trans_b else x2
    out = torch.matmul(x1.float(), x2_eff.float())
    if bias is not None:
        out = out + bias.float()
    # Single-process fallback assumes all ranks have identical seeded inputs.
    return out * int(world_size)


def mc2_distributed_golden(ctx, inputs, attrs):
    """Distributed HCCL baseline for MatmulAllReduce.

    Use torch.addmm for fused matmul+bias to match the MC2 kernel's
    cube-unit behavior (bias added in fp32 accumulator before truncation).
    """
    x2 = inputs["x2"]
    x2_eff = x2.t() if bool(attrs.get("is_trans_b", False)) else x2
    bias = inputs.get("bias")
    if bias is not None:
        out = torch.addmm(bias, inputs["x1"], x2_eff)
    else:
        out = torch.matmul(inputs["x1"], x2_eff)
    _all_reduce_sum(ctx, out, attrs)
    return out


def _all_reduce_sum(ctx, tensor: torch.Tensor, attrs):
    reduce_op = str(attrs.get("reduce_op", "sum")).lower()
    if reduce_op != "sum":
        raise ValueError(f"Unsupported reduce_op: {reduce_op}")
    ctx["dist"].all_reduce(tensor, op=ctx["dist"].ReduceOp.SUM)


def mc2_call_candidate(candidate, ctx, inputs, attrs):
    """Call task candidate with the MatmulAllReduce argument contract."""
    try:
        return candidate(
            inputs["x1"],
            inputs["x2"],
            ctx["hcomm_info"],
            ctx["world_size"],
            reduce_op=attrs.get("reduce_op", "sum"),
            bias=inputs.get("bias"),
            is_trans_b=bool(attrs.get("is_trans_b", False)),
        )
    except TypeError:
        return candidate(inputs["x1"], inputs["x2"], ctx["hcomm_info"], ctx["world_size"],
                         attrs.get("reduce_op", "sum"), inputs.get("bias"),
                         bool(attrs.get("is_trans_b", False)))
