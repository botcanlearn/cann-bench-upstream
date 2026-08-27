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
PointsToImage算子Torch Golden参考实现

将三维点通过3x4相机投影矩阵投影到二维图像平面。
仅执行齐次坐标投影，不做图像边界裁剪和有效性过滤。
"""
def points_to_image(
    points_3d: torch.Tensor,
    proj_matrix: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    将三维点投影为二维像素坐标

    Args:
        points_3d: 三维点坐标，shape为(...,3)，最后一维为(x,y,z)
        proj_matrix: 投影矩阵，shape为(3,4)或(B,3,4)
        eps: 深度分母最小值，用于避免除零

    Returns:
        二维像素坐标，shape为(...,2)，最后一维为(u,v)
    """
    if points_3d.shape[-1] != 3:
        raise ValueError("points_3d last dim must be 3")
    if proj_matrix.dim() not in (2, 3):
        raise ValueError("proj_matrix must have shape (3,4) or (B,3,4)")
    if proj_matrix.shape[-2:] != (3, 4):
        raise ValueError("proj_matrix last two dims must be (3,4)")

    orig_shape = points_3d.shape[:-1]
    device = points_3d.device
    input_dtype = points_3d.dtype

    if input_dtype == torch.float16:
        compute_dtype = torch.float32
    else:
        compute_dtype = input_dtype

    points_compute = points_3d.to(compute_dtype)
    P = proj_matrix.to(device=device, dtype=compute_dtype)

    if P.dim() == 3:
        if points_compute.dim() < 2:
            raise ValueError("Batched proj_matrix requires points_3d with batch dim")
        if points_compute.shape[0] != P.shape[0]:
            raise ValueError("Batched proj_matrix requires points_3d with same batch dim as proj_matrix")

        batch_size = P.shape[0]
        pts = points_compute.reshape(batch_size, -1, 3)
        ones = torch.ones((batch_size, pts.shape[1], 1), device=device, dtype=compute_dtype)
        pts_h = torch.cat([pts, ones], dim=-1)
        pixels_h = pts_h @ P.transpose(-1, -2)
        depth = pixels_h[..., 2:3].clamp_min(float(eps))
        pixels = pixels_h[..., :2] / depth
        pixels = pixels.reshape(orig_shape + (2,))
    else:
        pts = points_compute.reshape(-1, 3)
        ones = torch.ones((pts.shape[0], 1), device=device, dtype=compute_dtype)
        pts_h = torch.cat([pts, ones], dim=1)
        pixels_h = pts_h @ P.t()
        depth = pixels_h[:, 2:3].clamp_min(float(eps))
        pixels = pixels_h[:, :2] / depth
        pixels = pixels.reshape(orig_shape + (2,))

    if input_dtype == torch.float16:
        return pixels.to(input_dtype)
    return pixels

