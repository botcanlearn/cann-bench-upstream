#!/usr/bin/python3
# coding=utf-8

import torch


def _causal_mask(sq: int, skv: int, device) -> torch.Tensor:
    q_pos = torch.arange(sq, device=device).unsqueeze(1)
    k_pos = torch.arange(skv, device=device).unsqueeze(0)
    return k_pos > (q_pos + skv - sq)


def multi_query_attention_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    if q.dim() != 4 or k.dim() != 3 or v.dim() != 3:
        raise ValueError("q must be [B,Sq,Hq,D], k/v must be [B,Skv,D]")
    if k.shape != v.shape:
        raise ValueError("k and v must have the same shape")
    batch, seqlen_q, _, headdim = q.shape
    if k.shape[0] != batch or k.shape[2] != headdim:
        raise ValueError("k/v batch and head dimension must match q")

    out_dtype = q.dtype
    q_f = q.float().transpose(1, 2)
    k_f = k.float().unsqueeze(1)
    v_f = v.float().unsqueeze(1)
    scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * (headdim ** -0.5)
    if causal:
        mask = _causal_mask(seqlen_q, k.shape[1], scores.device)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v_f).transpose(1, 2).contiguous()
    return out.to(out_dtype)
