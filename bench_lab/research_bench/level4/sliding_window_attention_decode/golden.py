#!/usr/bin/python3
# coding=utf-8

import torch


def _window_mask(sq: int, skv: int, window_size: int, causal: bool, device) -> torch.Tensor:
    q_pos = torch.arange(sq, device=device).unsqueeze(1) + skv - sq
    k_pos = torch.arange(skv, device=device).unsqueeze(0)
    too_old = k_pos <= (q_pos - window_size)
    if causal:
        too_new = k_pos > q_pos
        return too_old | too_new
    return too_old


def sliding_window_attention_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int = 128,
    causal: bool = True,
) -> torch.Tensor:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q/k/v must be [B,S,H,D]")
    if k.shape != v.shape:
        raise ValueError("k and v must have the same shape")
    batch, seqlen_q, heads, headdim = q.shape
    if k.shape[0] != batch or k.shape[2:] != (heads, headdim):
        raise ValueError("k/v batch/head dimensions must match q")

    out_dtype = q.dtype
    q_f = q.float().transpose(1, 2)
    k_f = k.float().transpose(1, 2)
    v_f = v.float().transpose(1, 2)
    scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * (headdim ** -0.5)
    mask = _window_mask(seqlen_q, k.shape[1], window_size, causal, scores.device)
    scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v_f).transpose(1, 2).contiguous()
    return out.to(out_dtype)
