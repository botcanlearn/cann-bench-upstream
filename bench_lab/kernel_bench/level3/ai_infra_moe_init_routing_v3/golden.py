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
from typing import Optional, Tuple, List
import random
import numpy
import numpy as np
import math

def ai_infra_moe_init_routing_v3(
    x: torch.Tensor,
    expert_idx: torch.Tensor,
    scale: [torch.Tensor] = None,
    offset: Optional[torch.Tensor] = None,
    active_num: int = -1,
    expert_capacity: int = -1,
    expert_num: int = -1,
    drop_pad_mode: int = 0,
    expert_tokens_num_type: int = 0,
    expert_tokens_num_flag: bool = False,
    quant_mode: int = -1,
    active_expert_range: Optional[List[int]] = None,
    row_idx_type: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    set_seed(0)
    x_dtype = x.dtype
    if x_dtype != torch.int8:
        x = x.to(torch.float64)
    else:
        x = x
    expert_idx = expert_idx
    if scale is not None:
        scale = scale
    if offset is not None:
        offset = offset
    expert_start = active_expert_range[0]
    expert_end = active_expert_range[1]
    num_rows = x.shape[0]
    h = x.shape[1]
    k = expert_idx.shape[1]

    expert_idx_in = expert_idx.clone().reshape(-1)
    actual_expert_total_num = int(torch.sum(
        (expert_idx >= expert_start) & (expert_idx < expert_end)).item())

    # print("cpu:",actual_expert_total_num)
    max_int32 = torch.iinfo(torch.int32).max
    expert_idx_in[expert_idx_in < expert_start] = max_int32
    sorted_expert_indices = torch.argsort(expert_idx_in, dim=-1, stable=True)
    sorted_expert_idx = expert_idx_in[sorted_expert_indices]

    if row_idx_type == 1:
        expanded_row_idx = sorted_expert_indices.clone()
    else:
        # gather
        expanded_row_idx = torch.ones(num_rows * k, dtype=torch.int32, device=x.device) * -1
        tmp_indices = torch.arange(actual_expert_total_num, dtype=torch.int32, device=x.device)
        expanded_row_idx[sorted_expert_indices[:actual_expert_total_num]] = tmp_indices

    # 计算直方图
    if not expert_tokens_num_flag:
        expert_tokens_count = None
    else:
        if drop_pad_mode == 0:
            if expert_tokens_num_type == 1:
                expert_tokens_count = torch.bincount(
                    sorted_expert_idx[:actual_expert_total_num] - expert_start,
                    minlength=(expert_end - expert_start))
            elif expert_tokens_num_type == 0:
                expert_tokens_count = torch.bincount(
                    sorted_expert_idx[:actual_expert_total_num] - expert_start,
                    minlength=(expert_end - expert_start))
                expert_tokens_count = torch.cumsum(expert_tokens_count, dim=0)
            elif expert_tokens_num_type == 2:
                # key-value
                unique_experts, counts = torch.unique(
                    sorted_expert_idx[:actual_expert_total_num], return_counts=True)
                expert_tokens_count = torch.stack([unique_experts.to(torch.int64), counts.to(torch.int64)], dim=1)
                pad_len = expert_num - expert_tokens_count.shape[0]
                if pad_len > 0:
                    pad_tensor = torch.zeros((pad_len, 2), dtype=torch.int64, device=x.device)
                    expert_tokens_count = torch.cat([expert_tokens_count, pad_tensor], dim=0)
        else:
            expert_tokens_count = torch.bincount(
                sorted_expert_idx[:actual_expert_total_num] - expert_start,
                minlength=(expert_end - expert_start))
        expert_tokens_count = expert_tokens_count.to(torch.int64)

    vaild_num = 0
    if drop_pad_mode == 0:
        if active_num <= 0:
            vaild_num = actual_expert_total_num
        else:
            vaild_num = min(active_num, actual_expert_total_num)
        expanded_scale = None
        expanded_x = x[sorted_expert_indices[:vaild_num] // k, :]
        if scale is not None and quant_mode == -1:
            expanded_scale = scale[sorted_expert_indices[:vaild_num] // k]
    else:
        # droppad=1时计算逻辑
        adapter_capacity(sorted_expert_indices, sorted_expert_idx, expert_capacity)

        sort_row_tmp = torch.full((expert_num * expert_capacity,), -1, dtype=torch.int64, device=x.device)
        offset_tmp = 0
        lastExpertId = 0
        for i in range(sorted_expert_indices.shape[0]):
            val = sorted_expert_indices[i].item()
            if val != -1:
                cur_expert = sorted_expert_idx[i].item()
                if lastExpertId != cur_expert:
                    offset_tmp = 0
                    lastExpertId = cur_expert
                sort_row_tmp[cur_expert * expert_capacity + offset_tmp] = val
                offset_tmp = offset_tmp + 1

        # expand_row_idx
        expanded_row_idx = torch.full(sorted_expert_indices.shape, -1, dtype=torch.int32, device=x.device)
        for i in range(sort_row_tmp.shape[0]):
            val = sort_row_tmp[i].item()
            if val != -1:
                expanded_row_idx[val] = i

        # expanded_x
        expanded_x_mask = torch.ones((expert_num * expert_capacity, h), dtype=torch.bool, device=x.device)
        expanded_x = torch.zeros((expert_num * expert_capacity, h), dtype=x.dtype, device=x.device)
        for i in range(sort_row_tmp.shape[0]):
            val = sort_row_tmp[i].item()
            if val != -1:
                expanded_x[i] = x[val // k]
                expanded_x_mask[i] = False

    # 非量化
    if quant_mode == -1:
        expanded_x = expanded_x
        expanded_row_idx = expanded_row_idx
        if scale is None or drop_pad_mode == 1:
            expanded_scale = None

    # 静态量化
    if quant_mode == 0:
        expanded_scale = None
        expanded_x_fp32 = expanded_x.to(torch.float32)
        scale_val = scale.to(torch.float32)
        offset_val = offset.to(torch.float32)
        scale_rst = expanded_x_fp32 * scale_val[0]
        add_offset = scale_rst + offset_val[0]
        round_data = torch.round(add_offset)
        round_data = torch.clamp(round_data, -128, 127)
        expanded_x = round_data.to(torch.int8)

    # 动态量化
    if quant_mode == 1:
        x_final = expanded_x.to(torch.float32)
        if scale is None:
            x_abs = torch.abs(x_final)
            x_max = torch.max(x_abs, dim=-1, keepdim=True)[0]
            expanded_scale = x_max / 127.0
            expanded_x = torch.round(x_final / expanded_scale).to(torch.int8)
        else:
            scale = scale.to(torch.float32)
            if scale.shape[0] == 1:
                x_final = x_final * scale
            else:
                if drop_pad_mode == 0:
                    x_final = x_final * scale[sorted_expert_idx[:vaild_num] - expert_start]
                else:
                    for i in range(sort_row_tmp.shape[0]):
                        val = sort_row_tmp[i].item()
                        if val != -1:
                            x_final[i] = x_final[i] * scale[i // expert_capacity]

            x_abs = torch.abs(x_final)
            x_max = torch.max(x_abs, dim=-1, keepdim=True)[0]
            expanded_scale = x_max / 127.0
            expanded_x = torch.round(x_final / expanded_scale).to(torch.int8)

    if drop_pad_mode == 1:
        expanded_x = expanded_x.masked_fill(expanded_x_mask, 0)
        expanded_x = expanded_x.reshape(expert_num, expert_capacity, h)

    if row_idx_type == 1:
        expanded_row_idx = expanded_row_idx[:vaild_num]

    if drop_pad_mode == 0:
        if expanded_scale is not None:
            expanded_scale = expanded_scale.flatten()[:vaild_num]
        if active_num <= 0:
            active_num = num_rows * k
        else:
            active_num = min(active_num, num_rows * k)
        expanded_x = expanded_x[:vaild_num]
        # 将张量转到 CPU 进行 padding，避免 NPU OOM
        original_device = expanded_x.device
        expanded_x, expanded_row_idx, expanded_scale = post_process_golden_output(
            expanded_x.cpu(),
            expanded_row_idx.cpu(),
            expanded_scale.cpu() if expanded_scale is not None else None,
            h, active_num, num_rows * k)
        expanded_x = expanded_x.to(original_device)
        expanded_row_idx = expanded_row_idx.to(original_device)
        if expanded_scale is not None:
            expanded_scale = expanded_scale.to(original_device)

    if expert_tokens_count is None:
        expert_tokens_count = torch.tensor([], dtype=torch.int64)
    else:
        expert_tokens_count = expert_tokens_count.to(torch.int64)
    if expanded_scale is None:
        # expanded_scale = torch.ones()
        expanded_scale = torch.tensor([], dtype=torch.float32)
    else:
        expanded_scale = expanded_scale.to(torch.float32).reshape(-1)

    expanded_row_idx = expanded_row_idx.to(torch.int32)
    if quant_mode == -1:
        expanded_x = expanded_x.to(x_dtype)
    return expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale  

def post_process_golden_output(expanded_x, expanded_row_idx, expanded_scale, h, active_num, totalLength):
    pad_x = torch.ones((active_num - expanded_x.shape[0], h), dtype=expanded_x.dtype, device=expanded_x.device)
    expanded_x = torch.cat([expanded_x, pad_x], dim=0)
    pad_idx = torch.full((totalLength - expanded_row_idx.shape[0],), -1, dtype=expanded_row_idx.dtype, device=expanded_row_idx.device)
    expanded_row_idx = torch.cat([expanded_row_idx, pad_idx], dim=0)
    if expanded_scale is not None:
        pad_scale = torch.ones((active_num - expanded_scale.shape[0],), dtype=expanded_scale.dtype, device=expanded_scale.device)
        expanded_scale = torch.cat([expanded_scale, pad_scale], dim=0)
    return expanded_x, expanded_row_idx, expanded_scale


def adapter_capacity(sorted_row_idx, sorted_expert_idx, capacity):
    count = 0
    last = sorted_expert_idx[0]
    for i, val in enumerate(sorted_expert_idx):
        if last != val:
            count = 1
            last = val
        else:
            count += 1
            if count > capacity:
                sorted_expert_idx[i] = -1
                sorted_row_idx[i] = -1

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
