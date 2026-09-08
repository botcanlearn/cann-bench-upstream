#!/usr/bin/python3
# coding=utf-8

import torch


def upsample_nearest2d(x: torch.Tensor, scale_h: int = 2, scale_w: int = 2) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError("x must be [B,C,H,W]")
    if scale_h <= 0 or scale_w <= 0:
        raise ValueError("scale_h/scale_w must be positive")
    y = x.repeat_interleave(scale_h, dim=2).repeat_interleave(scale_w, dim=3)
    return y.contiguous()
