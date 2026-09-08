# PagedAttentionKvCache 算子 API 描述

## 1. 算子简介

`PagedAttentionKvCache` 面向 LLM decode / prefill 场景，从分页 K/V cache 中按照 `page_table` 重建每个 batch 的有效上下文，并执行支持 GQA 的 attention。

**主要应用场景**：
- LLM paged attention decoding
- 多请求 KV cache block 管理
- GQA / MQA 推理中的 KV cache 复用

**算子特征**：
- 难度等级：L4（FusedComposite）
- 输入 `q` 为 `[B, Sq, Hq, D]`
- 输入 `k_cache` / `v_cache` 为 `[num_blocks, page_block_size, Hkv, D]`
- `page_table` 描述逻辑 block 到物理 block 的映射
- 支持 `Hq / Hkv` 的 GQA head 扩展

## 2. 算子定义

### 数学公式

对每个 batch `b`，先根据 `cache_seqlens[b]` 与 `page_table[b]` 从分页 cache 重建连续 KV：

```text
K_b = reconstruct(k_cache, cache_seqlens[b], page_table[b])
V_b = reconstruct(v_cache, cache_seqlens[b], page_table[b])
```

若 `Hq > Hkv`，每个 KV head 被 `Hq / Hkv` 个 query head 共享。随后计算：

```text
scores = Q_b @ K_b^T / sqrt(D)
Y_b = softmax(scores + causal_mask) @ V_b
```

`causal=True` 时使用右对齐 causal mask。对于 `Sq <= Skv`，第 `i` 个 query 只能访问 `j <= i + Skv - Sq` 的 key。

## 3. 接口规范

### 算子原型

```python
cann_bench.paged_attention_kv_cache(
    Tensor q,
    Tensor k_cache,
    Tensor v_cache,
    Tensor cache_seqlens,
    Tensor page_table,
    bool causal=True,
) -> Tensor output
```

### 输入参数说明

| 参数 | 类型 | Shape | dtype | 描述 |
|------|------|-------|-------|------|
| `q` | Tensor | `[B, Sq, Hq, D]` | float16 / bfloat16 / float32 | query 张量 |
| `k_cache` | Tensor | `[num_blocks, page_block_size, Hkv, D]` | 与 `q` 相同 | 分页 K cache |
| `v_cache` | Tensor | `[num_blocks, page_block_size, Hkv, D]` | 与 `q` 相同 | 分页 V cache |
| `cache_seqlens` | Tensor | `[B]` | int32 / int64 | 每个 batch 的有效 KV 长度 |
| `page_table` | Tensor | `[B, max_blocks_per_seq]` | int32 / int64 | 每个 batch 的逻辑页到物理页映射 |
| `causal` | bool | - | - | 是否应用右对齐 causal mask |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| `output` | `[B, Sq, Hq, D]` | 与 `q` 相同 | attention 输出 |

### 数据类型

| q/k_cache/v_cache dtype | cache_seqlens/page_table dtype | 输出 dtype |
|-------------------------|--------------------------------|-----------|
| float16 | int32 / int64 | float16 |
| bfloat16 | int32 / int64 | bfloat16 |
| float32 | int32 / int64 | float32 |

### 规则与约束

- `q` 必须为 4D，`k_cache` 与 `v_cache` 必须为 4D。
- `k_cache.shape == v_cache.shape`。
- `q.shape[-1] == k_cache.shape[-1] == v_cache.shape[-1]`。
- `Hq` 必须能被 `Hkv` 整除。
- `cache_seqlens[b]` 必须满足 `1 <= cache_seqlens[b] <= page_table.shape[1] * page_block_size`。
- `page_table` 中参与重建的物理 block id 必须位于 `[0, num_blocks)`。
- `causal=True` 时要求 `Sq <= cache_seqlens[b]`，避免 query 行全部被 mask。

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `B` | 1 ~ 8 | 覆盖单请求和小批量 decode |
| `Sq` | 1 ~ 128 | 覆盖 decode 与短 prefill |
| `num_blocks` | 8 ~ 128 | 公开 case 控制在 CPU golden 可执行范围内 |
| `page_block_size` | 8 / 16 / 32 | 常见 cache block 大小 |
| `Hq` | 4 ~ 32 | 必须能被 `Hkv` 整除 |
| `Hkv` | 1 ~ 16 | 覆盖 MQA / GQA / MHA |
| `D` | 32 / 64 / 128 | head dim |
| `causal` | {False, True} | 两者均覆盖 |

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


def _reconstruct_cache(cache: torch.Tensor, cache_seqlen: int, page_table_row: torch.Tensor) -> torch.Tensor:
    page_block_size = cache.shape[1]
    num_blocks = cache.shape[0]
    table_width = page_table_row.numel()
    max_len = table_width * page_block_size
    seq_len = max(1, min(int(cache_seqlen), max_len))
    blocks_needed = (seq_len + page_block_size - 1) // page_block_size

    parts = []
    for block_idx in range(blocks_needed):
        physical_block = int(page_table_row[block_idx].item()) % num_blocks
        start = block_idx * page_block_size
        remaining = seq_len - start
        take = min(page_block_size, remaining)
        parts.append(cache[physical_block, :take])
    return torch.cat(parts, dim=0)


def paged_attention_kv_cache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    page_table: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    if q.dim() != 4 or k_cache.dim() != 4 or v_cache.dim() != 4:
        raise ValueError("q, k_cache and v_cache must be 4D tensors")
    if k_cache.shape != v_cache.shape:
        raise ValueError("k_cache and v_cache must have the same shape")
    if cache_seqlens.dim() != 1 or page_table.dim() != 2:
        raise ValueError("cache_seqlens must be [B] and page_table must be [B, MaxBlocks]")

    batch, seqlen_q, nheads_q, headdim = q.shape
    if cache_seqlens.shape[0] != batch or page_table.shape[0] != batch:
        raise ValueError("cache_seqlens/page_table batch dimension must match q")
    nheads_kv = k_cache.shape[2]
    if k_cache.shape[-1] != headdim:
        raise ValueError("q and cache head dimensions must match")
    if nheads_q % nheads_kv != 0:
        raise ValueError("Hq must be divisible by Hkv")

    out_dtype = q.dtype
    scale = headdim ** -0.5
    group = nheads_q // nheads_kv
    outputs = []

    for b in range(batch):
        seq_len = int(cache_seqlens[b].item())
        if causal and seq_len < seqlen_q:
            seq_len = seqlen_q
        k = _reconstruct_cache(k_cache, seq_len, page_table[b])
        v = _reconstruct_cache(v_cache, seq_len, page_table[b])
        seq_len = k.shape[0]

        k = k.transpose(0, 1).unsqueeze(0).float()
        v = v.transpose(0, 1).unsqueeze(0).float()
        if group > 1:
            k = k.repeat_interleave(group, dim=1)
            v = v.repeat_interleave(group, dim=1)

        q_b = q[b:b + 1].transpose(1, 2).float()
        scores = torch.matmul(q_b, k.transpose(-2, -1)) * scale

        if causal:
            row_idx = torch.arange(seqlen_q, device=scores.device).unsqueeze(1)
            col_idx = torch.arange(seq_len, device=scores.device).unsqueeze(0)
            mask = col_idx > (row_idx + seq_len - seqlen_q)
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        out_b = torch.matmul(attn, v).transpose(1, 2).contiguous()
        outputs.append(out_b.to(out_dtype))

    return torch.cat(outputs, dim=0)
```
