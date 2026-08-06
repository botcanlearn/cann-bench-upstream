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


def get_input(
    x1: torch.Tensor,
    x2: torch.Tensor,
    scale: torch.Tensor,
    pertokenScaleOptional: torch.Tensor,
    groupList: torch.Tensor,
    sharedInput: torch.Tensor,
    logit: torch.Tensor,
    rowIndex: torch.Tensor,
    **attrs,
) -> list:
    """从 attrs.group_list_values / row_index_values 重建 groupList、rowIndex 张量。

    cases.yaml 将确定性的 cumsum 分组边界放在 group_list_values 属性、行索引放在
    row_index_values 属性（golden 读取它们），但被测 kernel 只看 groupList / rowIndex
    张量。若不由 get_input 重建，这两个张量会被 value_range 随机生成（可能为负、
    麻烦单调），导致 kernel 与 golden 分组/累加目标不一致。返回值同时替换 golden 与
    AI 算子的输入，确保对比公平。
    """
    gl = attrs.get("group_list_values")
    if gl is not None:
        groupList = torch.tensor(list(gl), dtype=torch.int64, device=x1.device)
    ri = attrs.get("row_index_values")
    if ri is not None:
        rowIndex = torch.tensor(list(ri), dtype=torch.int64, device=x1.device)
    return [x1, x2, scale, pertokenScaleOptional, groupList, sharedInput, logit, rowIndex]


def _cumsum_groups(groupList: torch.Tensor):
    """将 cumsum groupList 转换为各专家的 [start, end) 行区间。"""
    ends = [int(v) for v in groupList.detach().cpu().tolist()]
    starts = [0] + ends[:-1]
    return list(zip(starts, ends))


def grouped_matmul_finalize_routing(
    x1: torch.Tensor,
    x2: torch.Tensor,
    scale: torch.Tensor,
    pertokenScaleOptional: torch.Tensor,
    groupList: torch.Tensor,
    sharedInput: torch.Tensor,
    logit: torch.Tensor,
    rowIndex: torch.Tensor,
    output_bs: int,
    sharedInputWeight: float = 1.0,
    sharedInputOffset: int = 0,
) -> torch.Tensor:
    """AscendC 实现目标及 A2/A3 A8W8 per-token/per-channel 路径 Golden。

    对齐 aclnnGroupedMatmulFinalizeRoutingWeightNzV2 的以下固定路径：
    x1/x2 为 INT8，scale 为 FP32 per-channel，per-token scale 为 FP32，
    sharedInput 为 BF16，rowIndex 为 INT64，输出为 FP32；不含 bias、
    offset 和 antiquant，矩阵不转置，groupListType=0。

    x1 的 M 行是按专家连续排列的路由记录，cumsum groupList 划分各专家的行区间。对第 i 个专家：

        acc_i = MatMul_INT32(x1[start_i:end_i, :], x2[i, :, :])
        routed_i = FP32(acc_i) * scale[i, None, :]
                   * pertokenScaleOptional[start_i:end_i, None]
                   * logit[start_i:end_i, None]
        out[rowIndex[t], :] += routed[t, :]

    sharedInput 会先乘 sharedInputWeight，再从 sharedInputOffset 行开始加入out。
    sharedInputOffset 不参与 rowIndex 的 scatter-add 索引计算。logit由上游提供，本算子不执行 softmax 或归一化。

    输入约定：
        x1: [M, K], INT8
        x2: [E, K, N], INT8 逻辑矩阵；真实算子的 FRACTAL_NZ 转换由适配器负责
        scale: [E, N], FP32
        pertokenScaleOptional: [M], FP32
        groupList: [E], INT64 cumsum
        sharedInput: [L, N], BF16
        logit: [M], FP32
        rowIndex: [M], INT64，取值范围 [0, output_bs)

    输出为 [output_bs, N] FP32，精度路径为 INT8 x INT8 -> INT32 累加 -> FP32 反量化与聚合。
    """
    # x1 已按专家排序，groupList 给出各专家的累积结束位置。
    groups = _cumsum_groups(groupList)
    m = x1.shape[0]
    n = x2.shape[2]
    shared_len = sharedInput.shape[0]
    shared_end = int(sharedInputOffset) + shared_len

    route_scale = logit * pertokenScaleOptional

    routed = torch.empty((m, n), dtype=torch.float32, device=x1.device)
    for expert, (start, end) in enumerate(groups):
        # 显式转为 INT32，以表达 INT8 x INT8 的 INT32 累加。
        acc = torch.matmul(x1[start:end].to(torch.int32), x2[expert].to(torch.int32))
        expert_out = acc * scale[expert].reshape(1, n)
        routed[start:end] = expert_out * route_scale[start:end].reshape(-1, 1)

    # Kernel 对每条路由记录读取 rowIndex，并原子累加到对应输出行。
    out = torch.zeros((output_bs, n), dtype=torch.float32, device=x1.device)
    out[sharedInputOffset:shared_end] = sharedInput.to(torch.float32) * sharedInputWeight
    out.index_add_(0, rowIndex, routed)
    return out


if __name__ == "__main__":
    m, k, n, e = 1, 2048, 7168, 1
    result = grouped_matmul_finalize_routing(
        x1=torch.zeros((m, k), dtype=torch.int8),
        x2=torch.zeros((e, k, n), dtype=torch.int8),
        scale=torch.ones((e, n), dtype=torch.float32),
        pertokenScaleOptional=torch.ones((m,), dtype=torch.float32),
        groupList=torch.tensor([m], dtype=torch.int64),
        sharedInput=torch.zeros((m, n), dtype=torch.bfloat16),
        logit=torch.ones((m,), dtype=torch.float32),
        rowIndex=torch.tensor([0], dtype=torch.int64),
        output_bs=m,
        sharedInputWeight=1.0,
        sharedInputOffset=0
    )
    print(result.shape, result.dtype)
