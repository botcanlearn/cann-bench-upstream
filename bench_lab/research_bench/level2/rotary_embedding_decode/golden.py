#!/usr/bin/python3
# coding=utf-8

import torch


def _rotate(
    x: torch.Tensor,
    cos_pos: torch.Tensor,
    sin_pos: torch.Tensor,
    rotary_dim: int,
    interleaved: bool,
) -> torch.Tensor:
    out_dtype = x.dtype
    x_f = x.float()
    cos_f = cos_pos.float()
    sin_f = sin_pos.float()

    while cos_f.dim() < x_f.dim():
        cos_f = cos_f.unsqueeze(-2)
        sin_f = sin_f.unsqueeze(-2)

    x_rot = x_f[..., :rotary_dim]
    x_pass = x_f[..., rotary_dim:]
    half = rotary_dim // 2

    if interleaved:
        x_even = x_rot[..., 0::2]
        x_odd = x_rot[..., 1::2]
        y_even = x_even * cos_f - x_odd * sin_f
        y_odd = x_odd * cos_f + x_even * sin_f
        y_rot = torch.empty_like(x_rot)
        y_rot[..., 0::2] = y_even
        y_rot[..., 1::2] = y_odd
    else:
        x1 = x_rot[..., :half]
        x2 = x_rot[..., half:]
        y1 = x1 * cos_f - x2 * sin_f
        y2 = x2 * cos_f + x1 * sin_f
        y_rot = torch.cat([y1, y2], dim=-1)

    y = torch.cat([y_rot, x_pass], dim=-1)
    return y.to(out_dtype)


def rotary_embedding_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_dim: int = -1,
    interleaved: bool = False,
):
    """Apply rotary embedding to q/k according to position_ids."""
    if q.dim() != 4 or k.dim() != 4:
        raise ValueError("q and k must be 4D [B,S,H,D]")
    if q.shape[0] != k.shape[0] or q.shape[1] != k.shape[1] or q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have same B, S and D")
    if position_ids.shape != q.shape[:2]:
        raise ValueError(f"position_ids shape must be {q.shape[:2]}, got {tuple(position_ids.shape)}")

    head_dim = q.shape[-1]
    if rotary_dim == -1:
        rotary_dim = head_dim
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2 != 0:
        raise ValueError(f"rotary_dim must be positive, even and <= head_dim, got {rotary_dim}")
    if cos.shape != sin.shape or cos.shape[-1] != rotary_dim // 2:
        raise ValueError("cos/sin must have shape [MaxPos, rotary_dim/2]")

    pos = position_ids.to(torch.long)
    if pos.numel() and (pos.min() < 0 or pos.max() >= cos.shape[0]):
        raise ValueError("position_ids out of cos/sin table range")

    cos_pos = cos.index_select(0, pos.reshape(-1)).reshape(*pos.shape, rotary_dim // 2)
    sin_pos = sin.index_select(0, pos.reshape(-1)).reshape(*pos.shape, rotary_dim // 2)

    q_out = _rotate(q, cos_pos, sin_pos, rotary_dim, interleaved)
    k_out = _rotate(k, cos_pos, sin_pos, rotary_dim, interleaved)
    return q_out, k_out
