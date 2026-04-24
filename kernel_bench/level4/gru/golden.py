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
from typing import List, Optional, Tuple, Union

"""
GRU 算子 Torch Golden 参考实现

支持两种输入格式：
1. TensorList 格式：weight_ih/hh 为列表，每层/方向一个 tensor
2. 展平格式：weight_ih/hh 为单个 tensor，所有层/方向权重按行拼接

proto.yaml 定义使用 TensorList，但测试框架单元素列表会展开为单个 tensor。

NPU 注意事项：
- 从 CPU 创建 GRU 后 .to(npu) 会触发 DynamicGRU 算子（只支持 float16）
- 直接在 NPU 上创建权重可使用静态 GRU 算子（支持 float32）
"""

def gru(
    x: torch.Tensor,
    weight_ih: Union[List[torch.Tensor], torch.Tensor],
    weight_hh: Union[List[torch.Tensor], torch.Tensor],
    bias_ih: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
    bias_hh: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
    h0: Optional[torch.Tensor] = None,
    inputSize: int = 0,
    hiddenSize: int = 0,
    numLayers: int = 1,
    bias: bool = True,
    batchFirst: bool = False,
    dropout: float = 0.0,
    bidirectional: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GRU 前向计算

    Args:
        x: 输入序列 (S, B, inputSize) 或 (B, S, inputSize) if batch_first
        weight_ih: TensorList 或单个展平 tensor
        weight_hh: TensorList 或单个展平 tensor
        bias_ih: TensorList? 或单个展平 tensor
        bias_hh: TensorList? 或单个展平 tensor
        h0: 初始隐藏状态
        inputSize: 输入特征维度
        hiddenSize: 隐藏状态维度
        numLayers: 层数
        bias: 是否使用偏置
        batchFirst: 输入格式是否为 (batch, seq, feature)
        dropout: 层间 dropout
        bidirectional: 是否双向

    Returns:
        y: 输出序列
        hn: 最终隐藏状态
    """
    num_directions = 2 if bidirectional else 1
    gate_size = 3 * hiddenSize

    # 统一转换为列表格式处理
    weight_ih_list = _ensure_list(weight_ih)
    weight_hh_list = _ensure_list(weight_hh)
    bias_ih_list = _ensure_list(bias_ih) if bias else None
    bias_hh_list = _ensure_list(bias_hh) if bias else None

    input_dtype = x.dtype
    target_device = x.device

    # 创建 GRU 模块（不需要先创建 CPU 版本再移动）
    gru_layer = torch.nn.GRU(
        input_size=inputSize,
        hidden_size=hiddenSize,
        num_layers=numLayers,
        bias=False,  # 先创建无 bias 版本，后面根据需要添加
        batch_first=batchFirst,
        dropout=dropout if numLayers > 1 else 0.0,
        bidirectional=bidirectional
    )

    # 关键：直接在目标设备上创建权重，避免从 CPU 移动触发 DynamicGRU 算子
    # DynamicGRU 只支持 float16 + FRACTAL_Z format，静态 GRU 支持 float32
    for layer in range(numLayers):
        layer_input = inputSize if layer == 0 else hiddenSize * num_directions
        for d in range(num_directions):
            idx = layer * num_directions + d
            suffix = f"l{layer}" if d == 0 else f"l{layer}_reverse"

            # 直接在目标设备创建权重参数
            wi_data = weight_ih_list[idx][:gate_size, :layer_input].float()
            wh_data = weight_hh_list[idx][:gate_size, :hiddenSize].float()

            # 创建新的 Parameter 在目标设备上
            wi_param = torch.nn.Parameter(wi_data.to(target_device))
            wh_param = torch.nn.Parameter(wh_data.to(target_device))

            setattr(gru_layer, f'weight_ih_{suffix}', wi_param)
            setattr(gru_layer, f'weight_hh_{suffix}', wh_param)

            if bias and bias_ih_list is not None and bias_hh_list is not None:
                bi_data = bias_ih_list[idx][:gate_size].float()
                bh_data = bias_hh_list[idx][:gate_size].float()
                bi_param = torch.nn.Parameter(bi_data.to(target_device))
                bh_param = torch.nn.Parameter(bh_data.to(target_device))
                setattr(gru_layer, f'bias_ih_{suffix}', bi_param)
                setattr(gru_layer, f'bias_hh_{suffix}', bh_param)
            elif bias:
                # 有 bias 要求但没传入偏置，创建零偏置
                bi_param = torch.nn.Parameter(
                    torch.zeros(gate_size, dtype=torch.float32, device=target_device)
                )
                bh_param = torch.nn.Parameter(
                    torch.zeros(gate_size, dtype=torch.float32, device=target_device)
                )
                setattr(gru_layer, f'bias_ih_{suffix}', bi_param)
                setattr(gru_layer, f'bias_hh_{suffix}', bh_param)

    x_float = x.float()
    if h0 is None:
        batch_size = x.shape[1] if not batchFirst else x.shape[0]
        h0 = torch.zeros(numLayers * num_directions, batch_size, hiddenSize,
                         dtype=torch.float32, device=target_device)
    else:
        h0 = h0.float()

    y, hn = gru_layer(x_float, h0)
    y = y.to(input_dtype)
    hn = hn.to(input_dtype)

    return y, hn


def _ensure_list(val):
    """确保值为列表格式"""
    if val is None:
        return None
    if isinstance(val, list):
        return val
    # 单个 tensor -> 转换为单元素列表
    return [val]