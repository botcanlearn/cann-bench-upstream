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
BevPool算子Torch Golden参考实现

BEVPoolV2 风格视锥特征池化：对每个视锥点 i，
    bev_feat.flat[ranks_bev[i]] += depth.flat[ranks_depth[i]] * feat.flat2d[ranks_feat[i], :]
即 gather（深度权重 + 特征行）→ 逐通道乘 → 按 BEV 网格索引 scatter-add。
ranks_bev 无序且可重复，kernel 侧需处理写冲突（原子加或排序聚合）。
plain golden 以 fp32 累加（bench 语义）；bev_pool_oracle 跟随输入精度
（golden_precision=fp64_cpu 下即为 fp64 真值）。
"""


def _bev_pool_core(depth, feat, ranks_depth, ranks_feat, ranks_bev, bev_h, bev_w, compute_dtype):
    """核心计算：以 compute_dtype 精度执行 gather-乘-scatter-add。"""
    C = feat.shape[-1]
    depth_flat = depth.reshape(-1).to(compute_dtype)               # [N_cam*D*fH*fW]
    feat_flat = feat.reshape(-1, C).to(compute_dtype)              # [N_cam*fH*fW, C]

    # gather: 深度权重 [P] 与特征行 [P, C]
    w = depth_flat[ranks_depth.long()]                             # [P]
    v = feat_flat[ranks_feat.long()]                               # [P, C]
    contrib = w.unsqueeze(-1) * v                                  # [P, C]

    # scatter-add: 按 ranks_bev 累加到 BEV 网格
    bev = torch.zeros(bev_h * bev_w, C, dtype=compute_dtype, device=feat.device)
    bev.index_add_(0, ranks_bev.long(), contrib)
    return bev.reshape(bev_h, bev_w, C)


def bev_pool(
    depth: torch.Tensor,
    feat: torch.Tensor,
    ranks_depth: torch.Tensor,
    ranks_feat: torch.Tensor,
    ranks_bev: torch.Tensor,
    bev_h: int,
    bev_w: int,
) -> torch.Tensor:
    """
    BEVPoolV2 风格视锥特征池化（plain golden = bench：fp32 累加）

    Args:
        depth: [N_cam, D, fH, fW] 视锥深度分布预测, float16/float32
        feat: [N_cam, fH, fW, C] 相机图像特征 (channel-last), dtype 与 depth 一致
        ranks_depth: [P] depth 展平后的线性索引 (int32), 取值 [0, N_cam*D*fH*fW - 1]
        ranks_feat: [P] feat 展平为 [N_cam*fH*fW, C] 后的行索引 (int32), 取值 [0, N_cam*fH*fW - 1]
        ranks_bev: [P] BEV 网格线性索引 (int32), 取值 [0, bev_h*bev_w - 1], 无序可重复
        bev_h: BEV 网格高度
        bev_w: BEV 网格宽度

    Returns:
        bev_feat: [bev_h, bev_w, C] BEV 网格特征, dtype 与输入一致
    """
    bev = _bev_pool_core(depth, feat, ranks_depth, ranks_feat, ranks_bev, bev_h, bev_w, torch.float32)
    return bev.to(feat.dtype)


def bev_pool_oracle(
    depth: torch.Tensor,
    feat: torch.Tensor,
    ranks_depth: torch.Tensor,
    ranks_feat: torch.Tensor,
    ranks_bev: torch.Tensor,
    bev_h: int,
    bev_w: int,
) -> torch.Tensor:
    """Oracle (g)：dtype-agnostic，计算精度跟随输入（fp64_cpu 下为 fp64 真值），不硬编码 .float()。"""
    return _bev_pool_core(depth, feat, ranks_depth, ranks_feat, ranks_bev, bev_h, bev_w, feat.dtype)
