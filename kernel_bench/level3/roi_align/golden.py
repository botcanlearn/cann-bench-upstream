#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import torch

"""
ROIAlign 算子 Torch Golden 参考实现

池化层，用于非均匀输入尺寸的特征图
公式: y = roi_align(x, boxes, output_size)

优先使用 torchvision.ops.roi_align 官方实现，
torchvision 不可用时使用纯 Python fallback（对标 torchvision 源码逻辑）。
"""

try:
    from torchvision.ops import roi_align as _tv_roi_align
    HAS_TORCHVISION = True
except Exception:
    HAS_TORCHVISION = False


def _bilinear_interpolate(
    input, roi_batch_ind, y, x, ymask, xmask,
):
    """Bilinear interpolation，对标 torchvision _bilinear_interpolate。"""
    _, channels, height, width = input.size()
    y = y.clamp(min=0)
    x = x.clamp(min=0)
    y_low = y.int()
    x_low = x.int()
    y_high = torch.where(y_low >= height - 1, height - 1, y_low + 1)
    y_low = torch.where(y_low >= height - 1, height - 1, y_low)
    y = torch.where(y_low >= height - 1, y.to(input.dtype), y)
    x_high = torch.where(x_low >= width - 1, width - 1, x_low + 1)
    x_low = torch.where(x_low >= width - 1, width - 1, x_low)
    x = torch.where(x_low >= width - 1, x.to(input.dtype), x)
    ly = y - y_low
    lx = x - x_low
    hy = 1.0 - ly
    hx = 1.0 - lx

    def masked_index(y_idx, x_idx):
        if ymask is not None:
            assert xmask is not None
            y_idx = torch.where(ymask[:, None, :], y_idx, 0)
            x_idx = torch.where(xmask[:, None, :], x_idx, 0)
        return input[
            roi_batch_ind[:, None, None, None, None, None],
            torch.arange(channels, device=input.device)[None, :, None, None, None, None],
            y_idx[:, None, :, None, :, None],
            x_idx[:, None, None, :, None, :],
        ]

    v1 = masked_index(y_low, x_low)
    v2 = masked_index(y_low, x_high)
    v3 = masked_index(y_high, x_low)
    v4 = masked_index(y_high, x_high)

    def outer_prod(y_t, x_t):
        return y_t[:, None, :, None, :, None] * x_t[:, None, None, :, None, :]

    w1 = outer_prod(hy, hx)
    w2 = outer_prod(hy, lx)
    w3 = outer_prod(ly, hx)
    w4 = outer_prod(ly, lx)
    return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4


def _roi_align_fallback(
    x, boxes, outputHeight, outputWidth,
    spatial_scale, sampling_ratio, aligned,
):
    """纯 Python fallback，对标 torchvision.ops.roi_align 的纯 Python 参考实现。"""
    orig_dtype = x.dtype
    x_fp32 = x.float()
    boxes_fp32 = boxes.float()
    _, channels, height, width = x_fp32.size()
    roi_batch_ind = boxes_fp32[:, 0].long()
    offset = 0.5 if aligned else 0.0
    roi_start_w = boxes_fp32[:, 1] * spatial_scale - offset
    roi_start_h = boxes_fp32[:, 2] * spatial_scale - offset
    roi_end_w = boxes_fp32[:, 3] * spatial_scale - offset
    roi_end_h = boxes_fp32[:, 4] * spatial_scale - offset
    roi_width = roi_end_w - roi_start_w
    roi_height = roi_end_h - roi_start_h
    if not aligned:
        roi_width = roi_width.clamp(min=1.0)
        roi_height = roi_height.clamp(min=1.0)
    bin_size_h = roi_height / outputHeight
    bin_size_w = roi_width / outputWidth
    exact_sampling = sampling_ratio > 0
    roi_bin_grid_h = sampling_ratio if exact_sampling else torch.ceil(roi_height / outputHeight)
    roi_bin_grid_w = sampling_ratio if exact_sampling else torch.ceil(roi_width / outputWidth)
    if exact_sampling:
        count = max(roi_bin_grid_h * roi_bin_grid_w, 1)
        iy = torch.arange(roi_bin_grid_h, device=x.device)
        ix = torch.arange(roi_bin_grid_w, device=x.device)
        ymask = None
        xmask = None
    else:
        count = torch.clamp(roi_bin_grid_h * roi_bin_grid_w, min=1)
        iy = torch.arange(height, device=x.device)
        ix = torch.arange(width, device=x.device)
        ymask = iy[None, :] < roi_bin_grid_h[:, None]
        xmask = ix[None, :] < roi_bin_grid_w[:, None]

    def from_K(t):
        return t[:, None, None]

    y = (
        from_K(roi_start_h)
        + torch.arange(outputHeight, device=x.device)[None, :, None] * from_K(bin_size_h)
        + (iy[None, None, :] + 0.5).to(x_fp32.dtype) * from_K(bin_size_h / roi_bin_grid_h)
    )
    x_pos = (
        from_K(roi_start_w)
        + torch.arange(outputWidth, device=x.device)[None, :, None] * from_K(bin_size_w)
        + (ix[None, None, :] + 0.5).to(x_fp32.dtype) * from_K(bin_size_w / roi_bin_grid_w)
    )

    val = _bilinear_interpolate(x_fp32, roi_batch_ind, y, x_pos, ymask, xmask)

    if not exact_sampling:
        val = torch.where(ymask[:, None, None, None, :, None], val, 0)
        val = torch.where(xmask[:, None, None, None, None, :], val, 0)

    output = val.sum((-1, -2))
    if isinstance(count, torch.Tensor):
        output = output / count[:, None, None, None]
    else:
        output = output / count

    return output.to(orig_dtype)


def roi_align(
    x: torch.Tensor, boxes: torch.Tensor, outputHeight: int, outputWidth: int,
    spatial_scale: float = 1.0, sampling_ratio: int = -1, aligned: bool = False,
) -> torch.Tensor:
    """
    池化层，用于非均匀输入尺寸的特征图

    公式: y = roi_align(x, boxes, output_size)

    Args:
        x: 输入特征图 [N, C, H, W]
        boxes: ROI框 [K, 5] (batch_idx, x1, y1, x2, y2)
        outputHeight: 输出高度
        outputWidth: 输出宽度
        spatial_scale: 空间缩放因子
        sampling_ratio: 采样比率 (-1 或 0 表示自动计算)
        aligned: 是否对齐 (aligned=True 时 ROI 坐标偏移 -0.5 像素)

    Returns:
        输出张量 [K, C, outputHeight, outputWidth]
    """
    if HAS_TORCHVISION:
        return _tv_roi_align(
            x, boxes, (outputHeight, outputWidth),
            spatial_scale=spatial_scale,
            sampling_ratio=sampling_ratio,
            aligned=aligned,
        )
    return _roi_align_fallback(
        x, boxes, outputHeight, outputWidth,
        spatial_scale, sampling_ratio, aligned,
    )
