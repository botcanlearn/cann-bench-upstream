# KvCacheAppend 算子 API 描述

## 1. 算子简介

`KvCacheAppend` 面向 LLM decode 每步 K/V cache 写入，将本步生成的新 K/V token 按 `slot_mapping` 写入分页 cache。

**主要应用场景**：
- decode 阶段每步 KV cache append
- 多请求 batch 的乱序 slot 写入
- paged attention 前的 cache 更新

**算子特征**：
- 难度等级：L2（Transform）
- 输入 cache shape 为 `[num_blocks, page_block_size, Hkv, D]`
- 输入新 token shape 为 `[T, Hkv, D]`
- `slot_mapping[t]` 表示第 `t` 个 token 写入 flatten 后 cache 的 slot
- 输出更新后的 `(k_cache_out, v_cache_out)`

## 2. 算子定义

将 cache 的前两维视为线性 slot：

```text
flat_k = reshape(k_cache, [num_blocks * page_block_size, Hkv, D])
flat_v = reshape(v_cache, [num_blocks * page_block_size, Hkv, D])
flat_k[slot_mapping[t]] = new_k[t]
flat_v[slot_mapping[t]] = new_v[t]
```

若 `slot_mapping` 中存在重复 slot，按输入 token 顺序执行写入，后写入的 token 覆盖先写入的 token。

## 3. 接口规范

### 算子原型

```python
cann_bench.kv_cache_append(
    Tensor k_cache,
    Tensor v_cache,
    Tensor new_k,
    Tensor new_v,
    Tensor slot_mapping,
) -> (Tensor k_cache_out, Tensor v_cache_out)
```

### 输入参数说明

| 参数 | 类型 | Shape | dtype | 描述 |
|------|------|-------|-------|------|
| `k_cache` | Tensor | `[num_blocks, page_block_size, Hkv, D]` | float16 / bfloat16 / float32 | 原 K cache |
| `v_cache` | Tensor | `[num_blocks, page_block_size, Hkv, D]` | 与 `k_cache` 相同 | 原 V cache |
| `new_k` | Tensor | `[T, Hkv, D]` | 与 `k_cache` 相同 | 待写入 K token |
| `new_v` | Tensor | `[T, Hkv, D]` | 与 `k_cache` 相同 | 待写入 V token |
| `slot_mapping` | Tensor | `[T]` | int32 / int64 | 每个 token 对应的线性 cache slot |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| `k_cache_out` | 与 `k_cache` 相同 | 与 `k_cache` 相同 | 写入后的 K cache |
| `v_cache_out` | 与 `v_cache` 相同 | 与 `v_cache` 相同 | 写入后的 V cache |

### 数据类型

| cache/new token dtype | slot_mapping dtype | 输出 dtype |
|-----------------------|--------------------|-----------|
| float16 | int32 / int64 | float16 |
| bfloat16 | int32 / int64 | bfloat16 |
| float32 | int32 / int64 | float32 |

### 规则与约束

- `k_cache.shape == v_cache.shape`。
- `new_k.shape == new_v.shape == [T, Hkv, D]`。
- `new_k.shape[1:] == k_cache.shape[2:]`。
- `slot_mapping.shape == [T]`。
- `slot_mapping` 中有效 slot 范围为 `[0, num_blocks * page_block_size)`。
- 重复 slot 按 token 顺序覆盖。

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `num_blocks` | 4 ~ 128 | 覆盖小 cache 与中等 cache |
| `page_block_size` | 8 / 16 / 32 | 常见 block 大小 |
| `T` | 1 ~ 256 | 覆盖单 token decode 与 batch append |
| `Hkv` | 1 ~ 16 | KV head 数 |
| `D` | 32 / 64 / 128 | head dim |

## 4. 精度要求

采用[生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)进行验证。

**误差指标**：

1. 平均相对误差（MERE）：采样点中相对误差平均值

   $$
   \text{MERE} = \text{avg}(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

2. 最大相对误差（MARE）：采样点中相对误差最大值

   $$
   \text{MARE} = \max(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

**通过标准**：

| 数据类型 | FLOAT16 | BFLOAT16 | FLOAT32 | HiFLOAT32 | FLOAT8 E4M3 | FLOAT8 E5M2 |
|----------|---------|----------|---------|-----------|-------------|-------------|
| **通过阈值(Threshold)** | 2^-10 | 2^-7 | 2^-13 | 2^-11 | 2^-3 | 2^-2 |

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过。

## 5. 标准 Golden 代码

```python
#!/usr/bin/python3
# coding=utf-8

import torch


def kv_cache_append(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    if k_cache.dim() != 4 or v_cache.dim() != 4:
        raise ValueError("k_cache and v_cache must be 4D tensors")
    if new_k.dim() != 3 or new_v.dim() != 3:
        raise ValueError("new_k and new_v must be 3D tensors")
    if k_cache.shape != v_cache.shape:
        raise ValueError("k_cache and v_cache must have the same shape")
    if new_k.shape != new_v.shape:
        raise ValueError("new_k and new_v must have the same shape")
    if new_k.shape[1:] != k_cache.shape[2:]:
        raise ValueError("new token H/D dimensions must match cache")
    if slot_mapping.dim() != 1 or slot_mapping.shape[0] != new_k.shape[0]:
        raise ValueError("slot_mapping must be [T]")

    k_out = k_cache.clone()
    v_out = v_cache.clone()
    total_slots = k_cache.shape[0] * k_cache.shape[1]
    slots = torch.remainder(slot_mapping.to(torch.long), total_slots)

    flat_k = k_out.reshape(total_slots, *k_cache.shape[2:])
    flat_v = v_out.reshape(total_slots, *v_cache.shape[2:])
    for i in range(new_k.shape[0]):
        slot = int(slots[i].item())
        flat_k[slot] = new_k[i]
        flat_v[slot] = new_v[i]

    return k_out, v_out
```
