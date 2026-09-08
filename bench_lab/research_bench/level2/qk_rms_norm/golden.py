#!/usr/bin/python3
# coding=utf-8

import torch


def _rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    out_dtype = x.dtype
    x_f = x.float()
    gamma_f = gamma.float()
    variance = x_f.pow(2).mean(dim=-1, keepdim=True)
    y = x_f * torch.rsqrt(variance + eps)
    y = y * gamma_f
    return y.to(out_dtype)


def qk_rms_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    gamma_q: torch.Tensor,
    gamma_k: torch.Tensor,
    eps: float = 1e-6,
):
    """Q/K 双路 RMSNorm reference."""
    return _rms_norm(q, gamma_q, eps), _rms_norm(k, gamma_k, eps)
