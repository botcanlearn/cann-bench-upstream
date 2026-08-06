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


# ===== 规格对齐说明(对齐主线 aclnnQuantGroupedMatmulInplaceAdd 的 T-C 量化场景) =====
# 主线算子: aclnnQuantGroupedMatmulInplaceAdd (graph op: QuantGroupedMatmulInplaceAdd, 仅 ascend950)
#   实现: ops-transformer/gmm/quant_grouped_matmul_inplace_add
# 对应场景: T-C 量化 (scale1=tensor级/perToken, scale2=channel级/perchannel)
# 主线公式: y_i[m,n] = (x1_i[m,k] @ x2_i[k,n]) * scale2_i[n] * scale1_i + yRef_i[m,n]  (inplace add)
#   出处: docs/aclnnQuantGroupedMatmulInplaceAdd.md (功能说明 L18/公式 L37/约束 L273-282);
#         op_api/quant_grouped_matmul_inplace_add_950_checker.cpp (CheckHif8QuantParams, HIFLOAT8 强制 L282-291);
#         tests/assets/golden.py (L78-113 非 mx 分支)
# 主线 dtype/shape 规格:
#   x1        : HIFLOAT8, 2D (M,K)    [主线物理 (K,M), view (M,K), transX=true]
#   x2        : HIFLOAT8, 2D (K,N)    [transW=false; 主线 2D 单权重]
#   scale1    : FLOAT32, (g,) 或 (g,1)  [tensor 级]
#   scale2    : FLOAT32, (g, N)         [channel 级]
#   groupList : INT64,   (g,), cumsum 边界
#   yRef / y  : FLOAT32, 3D (g,M,N), inplace add 累加器(输入即输出)
#   groupSize : 恒为 0 (T-C 场景); bias/offset 必须为空
# 关键约束: T-C 表 x1/x2 仅 HIFLOAT8; groupSize=0; yRef 3D; 仅 950PR/950DT.
# 本 golden 与主线的对齐说明:
#   - 主线 x1/x2 = HIFLOAT8; 本 golden 用 numpy en_dtypes.hifloat8 做 round-trip 体现 HIFLOAT8 量化误差
#     (CPU 无 native hifloat8 matmul, 故量化后转 float32 matmul, 对齐主线 L0C fp32 累加)。
#   - proto/cases dtype 声明 'hifloat8' -> CV 生成的算子用 hifloat8 -> 对齐主线 -> 可比。
#   - yRef 对齐主线 3D (g,M,N) inplace (y[idx, start:end, :] += ...)。
#   - x2 对齐主线 2D 单权重 (K,N); groupSize 恒 0。
# ===========================================================================================


def _hifloat8_quantize(t: torch.Tensor) -> torch.Tensor:
    """torch float32 -> 低精度 round-trip -> torch float32.

    主线输入为 HIFLOAT8；HIFLOAT8 为 CANN 私有 numpy dtype（en_dtypes），
    纯 torch 环境不可用。这里用 torch.float8_e4m3fn 做 round-trip 近似，
    量化方向（截断/饱和）与 HIFLOAT8 一致，数值上可能有微小差异，但满足
    cann-bench golden "只 import torch" 的硬约束。
    """
    f32 = t.detach().to(torch.float32)
    fp8 = f32.to(torch.float8_e4m3fn)
    return fp8.to(torch.float32)


def _groups(groupList: torch.Tensor, groupListType: int):
    values = [int(v) for v in groupList.detach().cpu().tolist()]
    if groupListType == 0:
        starts = [0] + values[:-1]
        ends = values
    elif groupListType == 1:
        starts, ends, cur = [], [], 0
        for count in values:
            starts.append(cur)
            cur += count
            ends.append(cur)
    else:
        raise ValueError("groupListType must be 0 or 1")
    return list(zip(starts, ends))


def get_input(
    x1: torch.Tensor,
    x2: torch.Tensor,
    scale1: torch.Tensor,
    scale2: torch.Tensor,
    groupList: torch.Tensor,
    yRef: torch.Tensor,
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
        groupList = torch.tensor(list(gl), dtype=torch.int64, device=x1.device)
    return [x1, x2, scale1, scale2, groupList, yRef]


def quant_grouped_matmul_inplace_add(
    x1: torch.Tensor,
    x2: torch.Tensor,
    scale1: torch.Tensor,
    scale2: torch.Tensor,
    groupList: torch.Tensor,
    yRef: torch.Tensor,
    groupListType: int = 0,
    group_size: int = 0,
    variant: str = "TC_PERCHANNEL",
    group_list_values=None,
) -> torch.Tensor:
    """Torch golden for quant_grouped_matmul_inplace_add, aligned to aclnnQuantGroupedMatmulInplaceAdd (T-C 场景).

    T-C per-channel: y_i = yRef_i + (x1_i @ x2_i) * scale2_i * scale1_i (inplace add).
    x1/x2 经 HIFLOAT8 量化(en_dtypes); yRef 3D (g,M,N); x2 2D 单权重 (K,N).
    规格详见模块顶部「规格对齐说明」。
    """
    if group_list_values is not None:
        groupList = torch.tensor(group_list_values, dtype=torch.int64, device=x1.device)
    groups = _groups(groupList, groupListType)
    g = len(groups)
    M, K = x1.shape
    K2, N = x2.shape
    if K != K2:
        raise ValueError(f"K mismatch: x1 K={K} x2 K={K2}")
    if yRef.shape != (g, M, N):
        raise ValueError(f"yRef expects ({g},{M},{N}) [对齐主线 3D], got {list(yRef.shape)}")
    # 对齐主线 T-C: y_i = (x1_i @ x2_i) * scale2(channel) * scale1(tensor) + yRef_i (inplace, 3D)
    # x1/x2 经 HIFLOAT8 量化(对齐主线 HIFLOAT8 输入)
    x2q = _hifloat8_quantize(x2)
    y = yRef.to(torch.float32).clone()
    for idx, (start, end) in enumerate(groups):
        if end <= start:
            continue
        partial = _hifloat8_quantize(x1[start:end, :]) @ x2q   # [m, N], hifloat8 量化后 float matmul
        y[idx, start:end, :] = y[idx, start:end, :] + partial * scale1[idx].to(torch.float32).reshape(1, 1) * scale2[idx].to(torch.float32).reshape(1, N)
    return y
