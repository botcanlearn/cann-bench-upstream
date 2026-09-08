#!/usr/bin/python3
# coding=utf-8

import torch


def ada_layer_norm_zero(
    x: torch.Tensor,
    cond: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    proj_weight: torch.Tensor,
    proj_bias: torch.Tensor = None,
    eps: float = 1e-6,
):
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    if x.dim() != 3 or cond.dim() != 2:
        raise ValueError("x must be [B,T,D] and cond must be [B,C]")
    batch, _, hidden = x.shape
    if cond.shape[0] != batch:
        raise ValueError("cond batch must match x batch")
    if proj_weight.shape[0] != 3 * hidden:
        raise ValueError("proj_weight first dim must be 3 * hidden")

    out_dtype = x.dtype
    normed = torch.nn.functional.layer_norm(x.float(), (hidden,), ln_weight.float(), ln_bias.float(), eps)
    params = torch.nn.functional.linear(cond.float(), proj_weight.float(), None if proj_bias is None else proj_bias.float())
    shift, scale, gate = params.chunk(3, dim=-1)
    y = normed * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    return y.to(out_dtype), gate.to(out_dtype)
