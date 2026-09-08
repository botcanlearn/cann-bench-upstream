#!/usr/bin/python3
# coding=utf-8

import torch


def _right_aligned_causal_mask(sq: int, skv: int, device) -> torch.Tensor:
    q_pos = torch.arange(sq, device=device).unsqueeze(-1)
    k_pos = torch.arange(skv, device=device).unsqueeze(0)
    offset = skv - sq
    return k_pos <= (q_pos + offset)


def causal_softmax_mask(
    scores: torch.Tensor,
    attention_mask: torch.Tensor = None,
    scale: float = 1.0,
    causal: bool = True,
) -> torch.Tensor:
    """Scale + mask + softmax reference for decoder attention scores."""
    if scores.dim() != 4:
        raise ValueError(f"scores must be 4D [B,H,Sq,Skv], got rank {scores.dim()}")

    out_dtype = scores.dtype
    x = scores.float() * float(scale)
    sq, skv = scores.shape[-2], scores.shape[-1]

    if causal:
        keep = _right_aligned_causal_mask(sq, skv, scores.device)
        x = x.masked_fill(~keep.view(1, 1, sq, skv), float("-inf"))

    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            x = x.masked_fill(~attention_mask, float("-inf"))
        else:
            x = x + attention_mask.float()

    y = torch.softmax(x, dim=-1)
    return y.to(out_dtype)
