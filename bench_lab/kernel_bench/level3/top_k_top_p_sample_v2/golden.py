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
"""
TopKTopPSampleV2 算子 Torch Golden 参考实现

公式: top_k_top_p_sample_v2(...)
"""

import torch
import numpy as np


FLT_NEG_INF = float('-inf')
USE_FAST_PROBS = True
ALL_P_MAX = 1.0


NP_TO_TORCH_DTYPE = {
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.uint8: torch.uint8,
    np.bool_: torch.bool,
}


def onlySoftmax(x, dim=-1):
    if dim < 0:
        dim = x.dim() + dim

    max_vals = torch.max(x, dim=dim, keepdim=True)[0]
    shifted = x - max_vals
    exp_vals = torch.exp(shifted)
    softmax_output = exp_vals / torch.sum(exp_vals, dim=dim, keepdim=True)
    return softmax_output


def _to_device_tensor(t, device):
    """将 numpy/torch 输入转换到目标设备。"""
    if t is None:
        return None
    if isinstance(t, torch.Tensor):
        return t.to(device)
    # numpy 或标量
    return torch.as_tensor(t).to(device)


def top_k_top_p_sample_v2(
    logits: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    q: torch.Tensor,
    min_ps: torch.Tensor,
    eps: float = 1e-8,
    is_need_logits: bool = False,
    top_k_guess: int = 32,
    ks_max: int = 1024,
    input_is_logits: bool = True,
    is_need_sample_result: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    TopKTopPSampleV2 Golden 参考实现。

    对齐真实算子接口（无 post_sample 属性）：
    - q 存在时走 qSample
    - q 不存在时走 argmax
    返回 (logits_select_idx, logits_top_kp_select, logits_idx, logits_sort_masked)。
    """
    device = logits.device
    logits = _to_device_tensor(logits, device)
    topK = _to_device_tensor(top_k, device)
    topP = _to_device_tensor(top_p, device)
    q = _to_device_tensor(q, device) if q is not None else None
    min_ps = _to_device_tensor(min_ps, device) if min_ps is not None else None

    batch_size, vocab_size = logits.shape

    # 计算实际 k_max（与 kernel 对齐）
    k_max_aligned = (ks_max * 4 + 32 - 1) // 32 * 32 // 4
    k_max = min(k_max_aligned, 1024)

    # 初始化结果张量
    rs_index = torch.zeros(batch_size, dtype=torch.long, device=device)
    logits_idx = torch.zeros((batch_size, vocab_size), dtype=torch.long, device=device)
    logits_sort_masked = torch.zeros((batch_size, vocab_size), dtype=torch.float32, device=device)

    # 根据是否需要 logits 初始化 rs_value
    if is_need_logits:
        if input_is_logits:
            rs_value = torch.ones((batch_size, vocab_size), dtype=torch.float32, device=device) * FLT_NEG_INF
        else:
            rs_value = torch.zeros((batch_size, vocab_size), dtype=torch.float32, device=device)
    else:
        rs_value = torch.empty(0, dtype=torch.float32, device=device)

    # compute golden
    for i in range(batch_size):
        original_logits = logits[i].float()

        k_val = topK[i].item()
        top_ks_max = min(k_max, vocab_size)
        use_top_k = (1 <= k_val <= top_ks_max)

        p = topP[i].item()
        use_top_p = p < ALL_P_MAX

        # 降序排序
        topk_logits, topk_indices = torch.sort(original_logits, dim=-1, descending=True, stable=True)

        # topK
        if use_top_k:
            k_val = min(k_val, vocab_size)
            topk_logits = topk_logits[:k_val]
            topk_indices = topk_indices[:k_val]

        # 归一化
        if input_is_logits:
            topk_probs = onlySoftmax(topk_logits, dim=-1)
        else:
            topk_probs = topk_logits

        # topP
        if use_top_p:
            sorted_probs, sorted_probs_indices = torch.sort(topk_probs, dim=-1, descending=True, stable=True)
            if p > 0:
                probs_sum = sorted_probs.cumsum(dim=-1)
                top_p_mask = (probs_sum - sorted_probs) >= p
            else:
                top_p_mask = torch.ones(sorted_probs.numel(), dtype=torch.bool)
                top_p_mask[0] = False

            top_p_sel = ~top_p_mask
            selected_probs_indices = sorted_probs_indices[top_p_sel]

            if USE_FAST_PROBS:
                selected_indices = topk_indices[selected_probs_indices]
                selected_logits = sorted_probs[top_p_sel]
            else:
                selected_indices = topk_indices[selected_probs_indices]
                selected_logits = topk_logits[selected_probs_indices]

            false_count = (top_p_sel > 0).sum().item()
        else:
            selected_indices = topk_indices
            selected_logits = topk_probs
            false_count = topk_probs.numel()
            top_p_sel = torch.ones(false_count, dtype=torch.bool)

        if p <= 0 and input_is_logits:
            selected_logits[0] = 1

        # minP
        if min_ps is not None:
            min_p = min_ps[i].item()
        else:
            min_p = -1

        if not use_top_k and not use_top_p and min_p < 1:
            selected_indices = torch.arange(len(original_logits), device=device)
            if input_is_logits:
                selected_logits = onlySoftmax(original_logits, dim=-1)
            else:
                selected_logits = original_logits

        if min_p <= 0:
            min_p_sel = torch.ones(false_count, dtype=torch.bool)
        elif min_p < 1:
            min_p_thd = torch.max(selected_logits) * min_p
            sel_prob_mask = selected_logits >= min_p_thd
            min_p_sel = sel_prob_mask
        else:
            min_p_sel = torch.zeros(false_count, dtype=torch.bool)
            min_p_sel[0] = True

        selected_logits = selected_logits[min_p_sel]
        selected_indices = selected_indices[min_p_sel]
        false_count = selected_logits.numel()

        if USE_FAST_PROBS:
            selected_probs = selected_logits
        else:
            if input_is_logits:
                selected_probs = onlySoftmax(selected_logits, dim=-1)
            else:
                selected_probs = selected_logits

        # 采样逻辑（真实算子语义）：q 存在用 qSample，否则 argmax
        if q is not None:
            q_i = q[i, :false_count]
            q_sample = selected_probs / (q_i.abs() + eps)
            probs_index = q_sample.argmax(dim=0).view(-1)
        else:
            probs_index = selected_probs.argmax(dim=0).view(-1)

        golden_index = selected_indices[probs_index].squeeze(0)
        rs_index[i] = golden_index

        if is_need_logits:
            rs_value[i, selected_indices] = original_logits[selected_indices]

    # 与自定义 kernel 的输出形状对齐：
    if not is_need_logits:
        rs_value = torch.zeros((batch_size, vocab_size), dtype=torch.float32, device=device)

    if not is_need_sample_result:
        logits_idx = torch.zeros((batch_size, vocab_size), dtype=torch.int64, device=device)
        logits_sort_masked = torch.zeros((batch_size, vocab_size), dtype=torch.float32, device=device)

    return rs_index, rs_value, logits_idx, logits_sort_masked
