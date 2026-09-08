#!/usr/bin/python3
# coding=utf-8

import torch


def patch_embedding_2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride_h: int = 16,
    stride_w: int = 16,
    pad_h: int = 0,
    pad_w: int = 0,
    flatten: bool = True,
) -> torch.Tensor:
    if x.dim() != 4 or weight.dim() != 4:
        raise ValueError("x and weight must be 4D tensors")
    if x.shape[1] != weight.shape[1]:
        raise ValueError("weight input channels must match x channels")
    if stride_h <= 0 or stride_w <= 0 or pad_h < 0 or pad_w < 0:
        raise ValueError("stride must be positive and padding must be non-negative")

    out_dtype = x.dtype
    y = torch.nn.functional.conv2d(
        x.float(),
        weight.float(),
        None if bias is None else bias.float(),
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
    )
    if flatten:
        y = y.flatten(2).transpose(1, 2).contiguous()
    return y.to(out_dtype)
