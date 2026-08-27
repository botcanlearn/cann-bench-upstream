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
PointsIn2DBoxesMask算子Torch Golden参考实现

判断二维图像平面点是否落入二维检测框内。
框格式为[x1, y1, x2, y2]，其中x2和y2为开区间边界。
"""
def points_in_2d_boxes_mask(
    points_2d: torch.Tensor,
    boxes_2d: torch.Tensor,
    image_width: int,
    image_height: int,
    use_image_bounds: bool = True,
) -> torch.Tensor:
    """
    判断二维点是否位于二维框内

    Args:
        points_2d: 二维点坐标，shape为(N,2)或(B,N,2)，最后一维为(u,v)
        boxes_2d: 二维框，shape为(4)、(K,4)、(B,4)或(B,K,4)
        image_width: 图像宽度
        image_height: 图像高度
        use_image_bounds: 是否同时要求点位于图像边界内

    Returns:
        bool掩码，shape为(N)、(K,N)、(B,N)或(B,K,N)
    """
    if points_2d.shape[-1] != 2:
        raise ValueError("points_2d last dim must be 2")
    if boxes_2d.shape[-1] != 4:
        raise ValueError("boxes_2d last dim must be 4")

    single_points = points_2d.dim() == 2
    if single_points:
        pts = points_2d.unsqueeze(0)
    else:
        pts = points_2d

    if pts.dim() != 3:
        raise ValueError("points_2d must have shape (N, 2) or (B, N, 2)")

    batch_size = pts.shape[0]
    u = pts[..., 0]
    v = pts[..., 1]

    boxes = boxes_2d.to(device=pts.device, dtype=pts.dtype)
    single_box = boxes.dim() == 1

    if single_box:
        boxes = boxes.view(1, 1, 4).expand(batch_size, 1, 4)
    elif boxes.dim() == 2:
        if boxes.shape[0] == batch_size and not single_points:
            boxes = boxes.view(batch_size, 1, 4)
            single_box = True
        else:
            boxes = boxes.unsqueeze(0).expand(batch_size, boxes.shape[0], 4)
    elif boxes.dim() == 3:
        if boxes.shape[0] != batch_size:
            raise ValueError("batched boxes_2d must share batch size with points_2d")
    else:
        raise ValueError("boxes_2d must have shape (4), (K,4), (B,4), or (B,K,4)")

    x1 = boxes[..., 0].unsqueeze(-1)
    y1 = boxes[..., 1].unsqueeze(-1)
    x2 = boxes[..., 2].unsqueeze(-1)
    y2 = boxes[..., 3].unsqueeze(-1)

    uu = u.unsqueeze(1)
    vv = v.unsqueeze(1)
    mask = (uu >= x1) & (uu < x2) & (vv >= y1) & (vv < y2)

    if bool(use_image_bounds):
        in_img = (
            (uu >= 0)
            & (uu < int(image_width))
            & (vv >= 0)
            & (vv < int(image_height))
        )
        mask = mask & in_img

    if single_box:
        mask = mask.squeeze(1)
    if single_points:
        mask = mask.squeeze(0)
    return mask
