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
ApplyAdamW 算子 Torch Golden 参考实现

AdamW 优化器实现，解耦权重衰减
公式:
    m_t = beta1 * m_{t-1} + (1 - beta1) * grad
    v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
    m_hat = m_t / (1 - beta1^t)
    v_hat = v_t / (1 - beta2^t)
    var_t = var_{t-1} - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * var_{t-1})

注：API 不含 timestep 参数，按单步独立调用处理，t=1，
故 bias_correction = 1 - beta^1 = 1 - beta
"""
def apply_adam_w(
    var: torch.Tensor,
    grad: torch.Tensor,
    m: torch.Tensor,
    v: torch.Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    epsilon: float = 1e-8,
    maximize: bool = False
) -> torch.Tensor:
    """
    AdamW 优化器实现，解耦权重衰减

    Args:
        var: 变量张量（需要优化的参数）
        grad: 梯度张量
        m: 一阶矩张量（动量）
        v: 二阶矩张量
        lr: 学习率
        beta1: 一阶矩估计的指数衰减率
        beta2: 二阶矩估计的指数衰减率
        weight_decay: 权重衰减系数（解耦）
        epsilon: 数值稳定常数
        maximize: 是否最大化目标函数

    Returns:
        更新后的变量
    """
    # 更新一阶矩（动量）
    m.mul_(beta1).add_(grad, alpha=1 - beta1)
    # 更新二阶矩
    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

    # 偏差修正（t=1）
    m_hat = m / (1 - beta1)
    v_hat = v / (1 - beta2)

    # 计算更新量
    update = m_hat / (v_hat.sqrt() + epsilon)

    # 解耦的权重衰减
    if weight_decay != 0:
        update.add_(var, alpha=weight_decay)

    # 应用更新
    y = var + lr * update if maximize else var - lr * update
    return y
