#!/usr/bin/python3
# coding=utf-8

import torch


def _partition_windows(x: torch.Tensor, window_h: int, window_w: int):
    batch, height, width, channels = x.shape
    pad_h = (window_h - height % window_h) % window_h
    pad_w = (window_w - width % window_w) % window_w
    x_pad = torch.nn.functional.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    hp, wp = height + pad_h, width + pad_w
    windows = x_pad.reshape(batch, hp // window_h, window_h, wp // window_w, window_w, channels)
    windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.reshape(-1, window_h * window_w, channels), hp, wp


def _reverse_windows(windows: torch.Tensor, batch: int, height: int, width: int, hp: int, wp: int,
                     window_h: int, window_w: int) -> torch.Tensor:
    channels = windows.shape[-1]
    x = windows.reshape(batch, hp // window_h, wp // window_w, window_h, window_w, channels)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(batch, hp, wp, channels)
    return x[:, :height, :width, :].contiguous()


def window_attention_2d(
    x: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor = None,
    position_bias: torch.Tensor = None,
    window_h: int = 7,
    window_w: int = 7,
    num_heads: int = 4,
) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError("x must be [B, H, W, C]")
    if window_h <= 0 or window_w <= 0 or num_heads <= 0:
        raise ValueError("window_h/window_w/num_heads must be positive")
    batch, height, width, channels = x.shape
    if channels % num_heads != 0:
        raise ValueError("channels must be divisible by num_heads")
    if qkv_weight.shape != (3 * channels, channels):
        raise ValueError("qkv_weight must be [3C, C]")
    if qkv_bias is not None and qkv_bias.shape != (3 * channels,):
        raise ValueError("qkv_bias must be [3C]")

    out_dtype = x.dtype
    windows, hp, wp = _partition_windows(x, window_h, window_w)
    qkv = torch.nn.functional.linear(windows.float(), qkv_weight.float(), None if qkv_bias is None else qkv_bias.float())
    num_windows, tokens, _ = qkv.shape
    head_dim = channels // num_heads
    qkv = qkv.reshape(num_windows, tokens, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    scores = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)
    if position_bias is not None:
        if position_bias.shape != (num_heads, tokens, tokens):
            raise ValueError("position_bias must be [num_heads, window_h*window_w, window_h*window_w]")
        scores = scores + position_bias.float().unsqueeze(0)
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v).transpose(1, 2).reshape(num_windows, tokens, channels)
    return _reverse_windows(out.to(out_dtype), batch, height, width, hp, wp, window_h, window_w)
