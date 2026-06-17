#!/usr/bin/python3
# coding=utf-8

import torch
from typing import Optional, Union, Tuple


def all_gather_matmul(
    x1: torch.Tensor,
    x2: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    hcomm_info: str = "",
    world_size: int = 1,
    gather_output: bool = True,
    is_trans_b: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Single-process CPU reference for the MC2 all-gather matmul task.

    The real benchmark path for cases with ``mc2_distributed: true`` uses
    ``MC2DistributedEvaluator`` and HCCL. This function keeps the task usable
    for metadata inspection and golden-only smoke checks.
    """
    del hcomm_info
    x2_eff = x2.t() if is_trans_b else x2
    gathered = torch.cat([x1] * int(world_size), dim=0)
    out = torch.matmul(gathered.float(), x2_eff.float())
    if bias is not None:
        out = out + bias.float()
    out = out.to(x1.dtype)
    if gather_output:
        return out, gathered
    return out


def mc2_distributed_golden(ctx, inputs, attrs):
    """Distributed golden hook used by MC2DistributedEvaluator."""
    dist = ctx["dist"]
    x1 = inputs["x1"]
    x2 = inputs["x2"]
    bias = inputs.get("bias")
    world_size = int(ctx["world_size"])
    gather_shape = [x1.shape[0] * world_size] + list(x1.shape[1:])
    gathered = torch.empty(gather_shape, dtype=x1.dtype, device=x1.device)
    dist._all_gather_base(gathered, x1)
    x2_eff = x2.t() if bool(attrs.get("is_trans_b", False)) else x2
    out = torch.matmul(gathered, x2_eff)
    if bias is not None:
        out = out + bias
    if bool(attrs.get("gather_output", False)):
        return out, gathered
    return out
