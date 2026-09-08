#!/usr/bin/python3
# coding=utf-8

import torch


def downsample_2d(
    x: torch.Tensor,
    kernel_h: int = 2,
    kernel_w: int = 2,
    stride_h: int = 2,
    stride_w: int = 2,
    pad_h: int = 0,
    pad_w: int = 0,
) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError("x must be [B,C,H,W]")
    if min(kernel_h, kernel_w, stride_h, stride_w) <= 0 or min(pad_h, pad_w) < 0:
        raise ValueError("kernel/stride must be positive and padding must be non-negative")
    out_dtype = x.dtype
    y = torch.nn.functional.avg_pool2d(
        x.float(), kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w), padding=(pad_h, pad_w)
    )
    return y.to(out_dtype)
