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

import math

import torch


def flat_quant(x, p1, p2, clipRatio):
    """执行 BF16 per-token INT4 FlatQuant。

    计算语义：
        x1 = x @ p2
        x2 = p1 @ x1
        quantScale[k] = max(abs(x2[k, :, :])) * clipRatio / 7
        out = round(x2 / quantScale)，并裁剪到有符号 INT4 范围 [-8, 7]
        若非零 token 的 quantScale 下溢为 0，正值输出 7，负值输出 -8，零值输出 0。

    输入：
        x:
            shape 为 [K, M, N]、dtype 为 torch.bfloat16 的非空 Tensor。
            K <= 262144，M <= 256，N <= 256；INT4 输出要求 N 为偶数。
        p1:
            shape 为 [M, M]、dtype 为 torch.bfloat16 的非空 Tensor。
        p2:
            shape 为 [N, N]、dtype 为 torch.bfloat16 的非空 Tensor。
        clipRatio:
            范围为 (0, 1] 的有限 Host double 浮点数；None 在 Golden
            中按 1.0 处理。非有限值或超出范围时抛出 ValueError。

    输出：
        out:
            逻辑 shape 为 [K, M, N]，数值范围为 [-8, 7]。
            算子真实输出必须使用有符号 INT4。当前 PyTorch Golden 使用
            torch.int8 Tensor 承载逻辑 INT4 数值；真实 INT4 的物理存储
            和解包由算子运行及比较环节处理。
        quantScale:
            shape 为 [K]、dtype 为 torch.float32 的 per-token 量化因子。
            全零 token 的 scale 为 0；非零 token 的 scale 也可能因
            FP32 下溢而变为 0，此时 out 按上述符号饱和规则输出。

    典型 case（x、p1、p2 均为 torch.bfloat16）：
        - Smoke：x=[1, 2, 8]，p1=[2, 2]，p2=[8, 8]，clipRatio=1.0。
        - decode：x=[1, 64, 64]，p1=[64, 64]，p2=[64, 64]，
          clipRatio=1.0。
        - MoE/tail：x=[257, 86, 128]，p1=[86, 86]，
          p2=[128, 128]，clipRatio=0.5。

    完整测试集合见同目录 cases.csv。
    """
    clipRatio = 1.0 if clipRatio is None else float(clipRatio)
    if not math.isfinite(clipRatio) or not 0.0 < clipRatio <= 1.0:
        raise ValueError("clipRatio must be finite and in (0, 1]")

    # 输入 x 先右乘 p2，再由 p1 左乘。
    x1 = torch.matmul(x, p2)
    x2 = torch.matmul(p1, x1)

    # per-token 语义：每个 x2[k, :, :] 独立计算最大绝对值。
    x2_flat = x2.flatten(-2, -1)
    qscale = torch.abs(x2_flat).max(dim=-1, keepdim=True)[0].to(torch.float32)
    fp32 = torch.finfo(torch.float32)
    if clipRatio >= 8 * fp32.tiny:
        # 此范围内 7 / clipRatio 有限，保留已有 FP32 计算与舍入顺序。
        ratio = torch.ones_like(qscale) * 7 / clipRatio
        quantScale = torch.flatten(qscale / ratio)
    else:
        # clipRatio = mantissa * 2**exponent。避免先计算溢出的倒数，
        # 或将极小的 Host double 标量直接转换为 FP32 而过早变成 0。
        mantissa, exponent = math.frexp(clipRatio)
        scaled = qscale * (mantissa / 7.0)
        while exponent < 0:
            step = min(-exponent, 126)
            scaled = scaled * math.ldexp(1.0, -step)
            exponent += step
        quantScale = torch.flatten(scaled)

    # 公式：out = x2 / quantScale。
    # reshape 仅用于将 [K] 的 quantScale 广播到 [K, M, N]。
    scale = quantScale.reshape(x2.shape[0], 1, 1)

    # 先裁剪分子，再执行除法；与最终饱和到 [-8, 7] 等价，且不会
    # 因 x2 / scale 超出 FP32 范围而产生 Inf。零 scale 使用安全分母。
    positive_scale = scale > 0
    safe_scale = torch.where(positive_scale, scale, torch.ones_like(scale))
    lower = -8 * safe_scale.clamp(max=fp32.max / 8)
    upper = 7 * safe_scale
    bounded = torch.minimum(torch.maximum(x2.to(torch.float32), lower), upper)
    normalized = bounded / safe_scale

    # AscendC 内核使用 Cast(..., RoundMode::CAST_RINT, ...)：舍入到最近整数；
    # 当数值恰好位于两个整数的中点时取偶数（ties-to-even）。
    # 有符号 INT4 的表示范围为 [-8, 7]。
    # Golden 使用 INT8 承载逻辑 INT4；真实算子输出必须使用物理 INT4。
    rounded = torch.round(normalized).clamp(-8, 7)

    # FP32 scale 下溢不等于 token 全零。任何非零 BF16 值都远大于
    # 此时的 scale，量化后必然饱和；真正的零值仍输出 0。
    underflow_out = torch.where(x2 > 0, 7.0, torch.where(x2 < 0, -8.0, 0.0))
    out = torch.where(positive_scale, rounded, underflow_out).to(torch.int8)

    return out, quantScale
