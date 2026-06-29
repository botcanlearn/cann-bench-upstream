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
kv_rms_norm_rope_cache_v2 算子 Torch Golden 参考实现。

功能：对输入 kv 的尾轴拆分，左半边做 RMSNorm，右半边做 RoPE，然后将结果
      scatter 到 k_cache / ckv_cache 中，同时可选输出 k_rope 与 c_kv。
"""

import torch
import einops


# ============================================================
# 前置处理：规范化 index 输入并自动生成 index
# ============================================================

def get_input(
    kv: torch.Tensor,
    gamma: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    index: torch.Tensor,
    k_cache: torch.Tensor,
    ckv_cache: torch.Tensor,
    k_rope_scale: torch.Tensor = None,
    c_kv_scale: torch.Tensor = None,
    k_rope_offset: torch.Tensor = None,
    c_kv_offset: torch.Tensor = None,
    v: torch.Tensor = None,
    cache_mode: str = "Norm",
    **kwargs,
):
    """
    参数:
        kv:             输入 kv 张量 [Bkv, Nkv, Skv, Dkv]
        gamma:          RMSNorm 权重
        cos:            RoPE cos 张量
        sin:            RoPE sin 张量
        index:          索引张量，未提供或为空时自动生成
        k_cache:        K cache 参考张量
        ckv_cache:      V cache 参考张量
        k_rope_scale:   K 量化缩放
        c_kv_scale:     V 量化缩放
        k_rope_offset:  K 量化偏移
        c_kv_offset:    V 量化偏移
        v:              可选 V 输入
        cache_mode:     缓存模式

    返回:
        按 golden 函数签名顺序排列的输入张量元组。
    """

    kv_shape = kv.shape
    Bkv = kv_shape[0]
    Skv = kv_shape[2]

    g = torch.Generator()
    g.manual_seed(1)

    # ---- PA BLK 场景 ----
    if "BLK" in cache_mode:
        S = k_cache.shape[1]
        B = k_cache.shape[0]
        CeilDivS = (Skv + S - 1) // S
        data = list(range(0, Bkv * CeilDivS * S, S))
        index = torch.Tensor(data)
        index = index.to(torch.int64)

    # ---- PA 非 BLK 场景 ----
    elif "PA" in cache_mode:
        S = k_cache.shape[1]
        B = k_cache.shape[0]
        data = list(range(-1, B * S))
        data = torch.tensor(data)
        if Bkv * Skv > B * S:
            random_indices = torch.randperm(len(data), generator=g)
            sampled_data = data[random_indices[:B * S]]
            for _ in range(Bkv*Skv-B*S):
                    sampled_data.append(-1)
        else:
            random_indices = torch.randperm(len(data), generator=g)
            sampled_data = data[random_indices[:Bkv*Skv]]
        index = sampled_data
        index = index.to(torch.int64)

    # ---- Norm 场景 ----
    else:
        S = k_cache.shape[2]
        data = list(range(-1, S))
        data = torch.Tensor(data)
        index_list = []
        index = torch.zeros(Bkv, Skv).to(torch.int64)
        for i in range(Bkv):
            if Skv > S:
                random_indices = torch.randperm(len(data), generator=g)
                sampled_data = data[random_indices[:S]]
                for _ in range(Skv-S):
                    sampled_data.append(-1)
            else:
                random_indices = torch.randperm(len(data), generator=g)
                sampled_data = data[random_indices[:Skv]]
            sub_index = sampled_data.to(torch.int64)
            index[i,:] = sub_index

    return (
        kv, gamma, cos, sin, index, k_cache, ckv_cache,
        k_rope_scale, c_kv_scale, k_rope_offset, c_kv_offset, v,
    )


# ============================================================
# Golden 标杆计算函数
# ============================================================

def kv_rms_norm_rope_cache_v2(
    kv: torch.Tensor,
    gamma: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    index: torch.Tensor,
    k_cache: torch.Tensor,
    ckv_cache: torch.Tensor,
    k_rope_scale: torch.Tensor = None,
    c_kv_scale: torch.Tensor = None,
    k_rope_offset: torch.Tensor = None,
    c_kv_offset: torch.Tensor = None,
    v: torch.Tensor = None,
    epsilon: float = 1e-5,
    cache_mode: str = "Norm",
    is_output_kv: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    kv_rms_norm_rope_cache_v2 算子的 PyTorch Golden 实现。

    Args:
        kv: 输入 kv 张量，shape (Bkv, Nkv, Skv, Dkv)。
        gamma: RMSNorm 权重。
        cos: RoPE cos 张量。
        sin: RoPE sin 张量。
        index: 索引张量，不同 cache_mode 下 shape 不同。
        k_cache: K 缓存输出张量（in-place 更新）。
        ckv_cache: V 缓存输出张量（in-place 更新），对应算子实现中的 v_cache。
        k_rope_scale: K 量化缩放。
        c_kv_scale: V 量化缩放。
        k_rope_offset: K 量化偏移。
        c_kv_offset: V 量化偏移。
        v: 可选的 V 输入；为 None 时 method_mode=0，否则 method_mode=1。
        epsilon: RMSNorm 稳定系数。
        cache_mode: 缓存模式，如 "Norm", "PA", "PA_NZ", "PA_BLK_BNSD", "PA_BLK_NZ" 等。
        is_output_kv: 是否输出 kv。

    Returns:
        (k_cache, ckv_cache, k_rope, c_kv)
    """
    # 将 proto.yaml 参数名映射到原 golden 实现中的变量名
    from einops import rearrange

    v_cache = ckv_cache
    k_scale = k_rope_scale
    v_scale = c_kv_scale
    k_offset = k_rope_offset
    v_offset = c_kv_offset
    tensor_v = v
    eps = epsilon

    ori_dtype = kv.dtype
    ori_k_cache_dtype = k_cache.dtype
    ori_v_cache_dtype = v_cache.dtype

    kv_dtype = kv.dtype

    tensor_v = tensor_v.to(torch.float32)

    kv = kv.to(torch.float32)
    gamma = gamma.to(torch.float32)
    cos = cos.to(torch.float32)
    sin = sin.to(torch.float32)

    kv_shape = kv.shape
    Bkv = kv_shape[0]
    Nkv = kv_shape[1]
    Skv = kv_shape[2]
    Dkv = kv_shape[3]

    v_dim = tensor_v.shape[3]
    k_dim = Dkv

    # b n s d -> b s n d
    kv = rearrange(kv, 'b n s d -> b s n d')
    cos = rearrange(cos, 'b n s d -> b s n d')
    sin = rearrange(sin, 'b n s d -> b s n d')

    if cache_mode == "Norm" and k_scale is None and v_scale is None:
        is_output_kv = False

    rms_in = kv
    v_in = tensor_v
    v_in = rearrange(v_in, 'b n s d -> b s n d')

    # RMS Norm
    v = rms_in / torch.sqrt(torch.mean(rms_in ** 2, dim=-1, keepdim=True) + eps)
    v = v * gamma

    # RoPE
    rope_dim = cos.shape[-1]
    rope_in = v[..., :rope_dim]
    k = rope_in.view(Bkv, Skv, Nkv, rope_dim // 2, 2) \
                .transpose(-1, -2) \
                .reshape(Bkv, Skv, Nkv, rope_dim)
    k1 = k[..., : k.shape[-1] // 2]
    k2 = k[..., k.shape[-1] // 2 :]
    rotate_half_k = torch.cat((-k2, k1), dim=-1)
    k_embed = (k * cos) + (rotate_half_k * sin)

    kv_out = torch.cat([k_embed, v[..., rope_dim:]], dim=-1)
    k_embed_out = rearrange(kv_out, 'b s n d -> b n s d').to(kv_dtype)

    if k_scale is not None:
        kv_out = kv_out * k_scale
    if k_offset is not None:
        kv_out = kv_out + k_offset
    if k_scale is not None:
        kv_out = torch.round(kv_out).clamp(-128, 127)
    k_embed = kv_out

    # tensor v
    v_out = rearrange(v_in, 'b s n d -> b n s d').to(kv_dtype)
    if v_scale is not None:
        v_in = v_in * v_scale
    if v_offset is not None:
        v_in = v_in + v_offset
    if v_scale is not None:
        v_in = torch.round(v_in).clamp(-128, 127)
    v = v_in

    # ================================================================
    # Cache Update
    # ================================================================

    if cache_mode == "PA_BNSD" or cache_mode == "PA":
        k_cache_shape = k_cache.shape
        v_cache_shape = v_cache.shape
        # b n s d -> (b s) n d
        # 输入 k_cache/v_cache 为 BSND，直接展成 (b s) n d
        k_cache = rearrange(k_cache, 'b s n d -> (b s)  n d')
        v_cache = rearrange(v_cache, 'b s n d -> (b s)  n d')
        k_embed = rearrange(k_embed, 'b s n d -> (b s)  n d')
        v = rearrange(v, 'b s n d -> (b s) n d')
        for batch in range(len(index)):
            if index[batch] == -1:
                continue
            k_cache[index[batch], :, :] = k_embed[batch, :, :].to(k_cache.dtype)
            v_cache[index[batch], :, :] = v[batch, :, :].to(v_cache.dtype)
        k_cache = rearrange(k_cache, '(b s) n d -> b s n d', b=k_cache_shape[0])
        v_cache = rearrange(v_cache, '(b s) n d -> b s n d', b=v_cache_shape[0])
        k_cache = k_cache
        v_cache = v_cache

    elif cache_mode == "PA_NZ":
        k_cache_shape = k_cache.shape
        v_cache_shape = v_cache.shape
        bn = k_cache_shape[0]
        block_size = k_cache_shape[1]
        dk = k_cache_shape[-1]
        dv = v_cache_shape[-1]
        dk0 = 32 if k_cache.dtype == torch.int8 else 16
        dv0 = 32 if v_cache.dtype == torch.int8 else 16
        dk1 = dk // dk0
        dv1 = dv // dv0
        num_head = k_cache_shape[2]
        k_cache = k_cache.reshape(bn, num_head, dk1, block_size, dk0)
        v_cache = v_cache.reshape(bn, num_head, dv1, block_size, dv0)
        k_embed = rearrange(k_embed, 'b s n d -> (b s)  n d')
        v = rearrange(v, 'b s n d -> (b s) n d')
        for batch in range(len(index)):
            index_value = index[batch]
            if index_value < 0:
                continue
            bn_id = index_value // block_size
            block_offset = index_value % block_size
            for i in range(dk1):
                k_cache[bn_id, :, i, block_offset, :] = \
                    k_embed[batch, :, i * dk0:(i + 1) * dk0].to(k_cache.dtype)
            for i in range(dv1):
                v_cache[bn_id, :, i, block_offset, :] = \
                    v[batch, :, i * dv0:(i + 1) * dv0].to(v_cache.dtype)
        k_cache = k_cache.reshape(k_cache_shape)
        v_cache = v_cache.reshape(v_cache_shape)

    elif cache_mode == "PA_BLK_BNSD":
        k_cache_shape = k_cache.shape
        v_cache_shape = v_cache.shape
        block_size = k_cache_shape[1]
        block_num = k_cache_shape[0]
        ceil_div_s = (Skv + block_size - 1) // block_size
        for batch in range(Bkv):
            for seq_id in range(ceil_div_s):
                seq_start = seq_id * block_size
                seq_end = Skv if seq_id == (ceil_div_s - 1) else (seq_id + 1) * block_size
                copy_len = seq_end - seq_start
                index_value = index[batch * ceil_div_s + seq_id]
                cache_b = index_value // block_size
                if index_value == -1:
                    continue
                k_cache[cache_b, :copy_len, :, :] = \
                    k_embed[batch, seq_start:seq_end, :, :].to(k_cache.dtype)
                v_cache[cache_b, :copy_len, :, :] = \
                    v[batch, seq_start:seq_end, :, :].to(v_cache.dtype)

    elif cache_mode == "PA_BLK_NZ":
        k_cache_shape = k_cache.shape
        v_cache_shape = v_cache.shape
        bn = k_cache_shape[0]
        block_size = k_cache_shape[1]
        dk = k_cache_shape[-1]
        dv = v_cache_shape[-1]
        dk0 = 32 if k_cache.dtype == torch.int8 else 16
        dv0 = 32 if v_cache.dtype == torch.int8 else 16
        dk1 = dk // dk0
        dv1 = dv // dv0
        num_head = k_cache_shape[2]
        k_cache = k_cache.reshape(bn, num_head, dk1, block_size, dk0)
        v_cache = v_cache.reshape(bn, num_head, dv1, block_size, dv0)
        ceil_div_s = (Skv + block_size - 1) // block_size
        for batch in range(Bkv):
            for seq_id in range(ceil_div_s):
                seq_start = seq_id * block_size
                seq_end = Skv if seq_id == (ceil_div_s - 1) else (seq_id + 1) * block_size
                copy_len = seq_end - seq_start
                index_value = index[batch * ceil_div_s + seq_id]
                cache_b = index_value // block_size
                if index_value == -1:
                    continue
                for n_idx in range(num_head):
                    for i in range(dk1):
                        k_cache[cache_b, n_idx, i, :copy_len, :] = \
                            k_embed[batch, seq_start:seq_end, n_idx, i * dk0:(i + 1) * dk0].to(k_cache.dtype)
                    for i in range(dv1):
                        v_cache[cache_b, n_idx, i, :copy_len, :] = \
                            v[batch, seq_start:seq_end, n_idx, i * dv0:(i + 1) * dv0].to(v_cache.dtype)
        k_cache = k_cache.reshape(k_cache_shape)
        v_cache = v_cache.reshape(v_cache_shape)

    else:
        v_cache = rearrange(v_cache, 'b n s d -> b s n d')
        k_cache = rearrange(k_cache, 'b n s d -> b s n d')
        for batch in range(index.shape[0]):
            for sdx in range(index.shape[1]):
                if index[batch][sdx] == -1:
                    continue
                v_cache[batch, index[batch][sdx], :, :] = v[batch, sdx, :, :].to(v_cache.dtype)
                k_cache[batch, index[batch][sdx], :, :] = k_embed[batch, sdx, :, :].to(k_cache.dtype)
        v_cache = rearrange(v_cache, 'b s n d -> b n s d')
        k_cache = rearrange(k_cache, 'b s n d -> b n s d')

    # ================================================================
    # 输出打包
    # ================================================================
    if is_output_kv:
        output_data = (
            k_cache.to(ori_k_cache_dtype),
            v_cache.to(ori_v_cache_dtype),
            k_embed_out.to(ori_dtype),
            v_out.to(ori_dtype),
        )
    else:
        output_data = (
            k_cache.to(ori_k_cache_dtype),
            v_cache.to(ori_v_cache_dtype),
            torch.zeros_like(k_embed_out).to(ori_dtype),
            torch.zeros_like(v_out).to(ori_dtype),
        )
    return output_data
