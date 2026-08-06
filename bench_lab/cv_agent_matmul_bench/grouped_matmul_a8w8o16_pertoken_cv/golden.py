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
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    groupList: torch.Tensor,
    perTokenScale: torch.Tensor,
    **attrs,
) -> list:
    """从 attrs.group_list_values 重建确定性的 groupList 张量。

    cases.yaml 将 cumsum 分组边界放在 group_list_values 属性里（golden 读取它），
    但被测 kernel 只看 groupList 张量。若不由 get_input 重建，groupList 会被
    value_range 随机生成（可能为负、非单调），导致 kernel 与 golden 分组不一致。
    返回值同时替换 golden 与 AI 算子的输入，确保对比公平。
    """
    gl = attrs.get("group_list_values")
    if gl is not None:
        groupList = torch.tensor(list(gl), dtype=torch.int64, device=x.device)
    return [x, weight, scale, groupList, perTokenScale]


# Atlas A3: grouped_matmul.cpp A8W8O16 -> GMM_CV_SPLIT_IMP(GMMQuantMixCoreCompute, GMMProcess).
def grouped_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    groupList: torch.Tensor,
    perTokenScale: torch.Tensor,
    group_list_values=None,
) -> torch.Tensor:
    """执行 Atlas A3 aclnnGroupedMatmulV5 的 A8W8O16 per-token grouped matmul。

    对齐的官方 kernel 路径：
        op_kernel/grouped_matmul.cpp
          -> GMM_QUANT_BF16
          -> GMM_CV_SPLIT_IMP(GMMQuantMixCoreCompute, GMMProcess, ...)

    计算语义：
        groupList 使用 cumsum 边界将 x 的 M 轴划分给 E 个 expert。
        对每个非空 expert 区间 [start, end)：
            accumulator = x[start:end].int32 @ weight[expert].int32
            dequantized = accumulator.float32 * scale[expert].float32
            output = dequantized * perTokenScale[start:end].float32
        最后将 FP32 中间结果转换为 BF16。

        该顺序对应 GMMQuantMixCoreCompute 中的 INT32 Cube 累加、
        AscendDequant、per-token FP32 Mul 和最终 Cast。本 benchmark 固定
        无 bias、无 offset、actType=0，不包含激活或动态输出量化。

    输入：
        x:
            shape 为 [M, K]、dtype 为 torch.int8 的 routed token。
            Atlas A3 路径要求 K < 65536。
        weight:
            shape 为 [E, K, N]、dtype 为 torch.int8 的 ND expert 权重；
            本 benchmark 固定不转置，且 1 <= E <= 1024、N < 65536。
        scale:
            shape 为 [E, N]、dtype 为 torch.bfloat16 的 per-expert
            per-channel 反量化因子。
        groupList:
            shape 为 [E]、dtype 为 torch.int64 的非负单调非递减
            cumsum 边界。官方算子允许最后一个值小于等于 M；本 benchmark
            固定为 M，以保证输出的每一行都有定义。
            相邻值可以相等，表示对应 expert 为空。
        perTokenScale:
            shape 为 [M]、dtype 为 torch.float32 的 per-token 因子。

    Benchmark 辅助参数：
        group_list_values:
            runner 用于构造确定性 groupList 的辅助值，不是 ACLNN 参数。
            为 None 时直接读取 groupList Tensor。

    固定场景：
        本函数直接表达单 Tensor 输出、M 轴分组、cumsum groupList 和
        无激活语义，因此不再暴露只负责选路的 ACLNN 属性。

    输出：
        shape 为 [M, N]、dtype 为 torch.bfloat16。最终转换与官方
        ST reference 一致，使用 PyTorch .to(torch.bfloat16) 表达。

    典型 case：
        - 常规：x=[16,128]，weight=[2,128,64]，groupList=[8,16]。
        - K/N tail：x=[18,1025]，weight=[4,1025,511]，
          groupList=[4,9,13,18]。

    完整测试集合见同目录 cases.yaml 和 cases.csv。
    """
    m = x.shape[0]
    n = weight.shape[2]
    groups = group_list_values
    if groups is None:
        groups = groupList.detach().to(device="cpu").tolist()

    out = torch.zeros(m, n, dtype=torch.float32, device=x.device)

    start = 0
    for expert_id, end in enumerate(groups):
        if end > start:
            accumulator = torch.matmul(
                x[start:end].to(torch.int32),
                weight[expert_id].to(torch.int32),
            )
            dequantized = accumulator.to(torch.float32) * scale[expert_id]
            out[start:end] = dequantized * perTokenScale[start:end, None]
        start = end

    return out.to(torch.bfloat16)
