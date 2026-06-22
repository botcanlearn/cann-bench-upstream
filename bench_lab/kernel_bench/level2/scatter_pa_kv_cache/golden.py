# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import torch
import copy


# ============================================================
# 前置处理：规范化 optional 输入并自动生成 slotMapping
# ============================================================

def get_input(
    key: torch.Tensor,
    key_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    value: torch.Tensor,
    value_cache: torch.Tensor,
    cache_mode: str = "PA_NZ",
    **kwargs,
):
    """
    算子参数前置处理。

    将 None 或空张量形式的 optional 输入统一为空张量，并在 slot_mapping 未提供时，
    根据 key/key_cache 形状自动生成 slot_mapping。

    参数:
        key:                输入 key 张量 [numTokens, numHeads, headSize]
        key_cache:          key cache 参考张量
        slot_mapping:       slot 索引张量 [numTokens]，未提供或为空时自动生成
        value:              输入 value 张量
        value_cache:        value cache 参考张量
        cache_mode:         cache 格式，PA_NZ 场景固定为 "PA_NZ"

    返回:
        按 scatter_pa_kv_cache 签名顺序排列的输入元组。
    """
    num_blocks = key_cache.shape[0]
    block_size = key_cache.shape[2]
    bs = key.shape[0]
    slot_mapping = torch.randperm(num_blocks * block_size)[:bs].to(torch.int32)

    return (
        key, key_cache, slot_mapping, value, value_cache,
        cache_mode,
    )


# ============================================================
# Golden 实现：PA_NZ 场景
# ============================================================

def scatter_pa_kv_cache(
    key: torch.Tensor,
    key_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    value: torch.Tensor,
    value_cache: torch.Tensor,
    cache_mode: str = "PA_NZ",
) -> tuple:
    """
    ScatterPaKvCache Golden 实现（PA_NZ 场景）。

    将输入的 key/value token 按 slot_mapping 写入 FRACTAL_NZ 语义排布的 KV Cache。
    PA_NZ 场景下不使用 compress_lens / compress_seq_offset / seq_lens。

    参数:
        key:                输入 key 张量 [numTokens, numHeads, kHeadSize]
        key_cache:          key cache 张量 [numBlocks, NH*kHS/lastDim, blockSize, lastDim]
        slot_mapping:       slot 索引张量 [numTokens]
        value:              输入 value 张量 [numTokens, numHeads, vHeadSize]
        value_cache:        value cache 张量 [numBlocks, NH*vHS/lastDim, blockSize, lastDim]
        cache_mode:         固定 "PA_NZ"

    返回:
        (key_cache, value_cache): 更新后的 cache 张量对
    """
    # clone cache 避免 in-place 修改输入，保证 golden 可重复执行且与 AI 输出隔离
    key_cache_golden = copy.deepcopy(key_cache)
    value_cache_golden = copy.deepcopy(value_cache)

    block_size = key_cache.shape[2]
    lastDim_k = key_cache.shape[3]
    lastDim_v = value_cache.shape[3]
    num_head = key.shape[1]
    k_head_size = key.shape[2]
    v_head_size = value.shape[2]

    for i, slot in enumerate(slot_mapping):
        if slot < 0:
            continue
        block_index = slot // block_size
        block_offset = slot % block_size

        token_key = key[i].reshape(num_head * k_head_size)
        for k in range(num_head * k_head_size // lastDim_k):
            key_cache_golden[block_index][k][block_offset][:] = token_key[k * lastDim_k: k * lastDim_k + lastDim_k]

        token_value = value[i].reshape(num_head * v_head_size)
        for v in range(num_head * v_head_size // lastDim_v):
            value_cache_golden[block_index][v][block_offset][:] = token_value[v * lastDim_v: v * lastDim_v + lastDim_v]

    return key_cache_golden, value_cache_golden
