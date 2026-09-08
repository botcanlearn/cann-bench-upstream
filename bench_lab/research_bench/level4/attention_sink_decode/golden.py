#!/usr/bin/python3
# coding=utf-8

import torch


def _sink_causal_mask(sq: int, skv: int, sink: int, device) -> torch.Tensor:
    q_pos = torch.arange(sq, device=device).unsqueeze(1)
    ordinary_pos = torch.arange(skv, device=device).unsqueeze(0)
    ordinary_mask = ordinary_pos > (q_pos + skv - sq)
    sink_mask = torch.zeros((sq, sink), dtype=torch.bool, device=device)
    return torch.cat((sink_mask, ordinary_mask), dim=1)


def attention_sink_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink_k: torch.Tensor,
    sink_v: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4 or sink_k.dim() != 4 or sink_v.dim() != 4:
        raise ValueError("all inputs must be 4D tensors")
    if k.shape != v.shape or sink_k.shape != sink_v.shape:
        raise ValueError("k/v and sink_k/sink_v must have matching shapes")
    batch, seqlen_q, heads, headdim = q.shape
    if k.shape[0] != batch or sink_k.shape[0] != batch or k.shape[2:] != (heads, headdim) or sink_k.shape[2:] != (heads, headdim):
        raise ValueError("batch/head dimensions must match")

    out_dtype = q.dtype
    kv_k = torch.cat((sink_k, k), dim=1)
    kv_v = torch.cat((sink_v, v), dim=1)
    q_f = q.float().transpose(1, 2)
    k_f = kv_k.float().transpose(1, 2)
    v_f = kv_v.float().transpose(1, 2)
    scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * (headdim ** -0.5)
    if causal:
        mask = _sink_causal_mask(seqlen_q, k.shape[1], sink_k.shape[1], scores.device)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v_f).transpose(1, 2).contiguous()
    return out.to(out_dtype)
