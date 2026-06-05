# PagedAttentionKvCache 算子 API 描述

## 1. 算子简介

Paged KV Cache Attention，从 paged K/V cache 中按 page_table 重建上下文，并支持 GQA head 扩展与 causal mask。

**主要应用场景**：

- LLM paged attention decoding
- 多请求 KV cache block 管理
- GQA + paged cache 推理

**算子特征**：

- k_cache/v_cache shape 为 [num_blocks,page_block_size,Hkv,D]
- cache_seqlens 给出每个 batch 有效上下文长度
- page_table 给出逻辑页到物理页映射
- 难度等级：L4（FusedComposite）

## 2. 算子定义

### 数学公式

```text
Reconstruct paged K/V cache through page_table and compute cache attention.
```

### Golden 语义来源

- 参考实现：`/root/asc/cv_agent_all/cv_agent/tile2asc/paged_attention_kv_cache/model.py`
- staging golden 只保留 schema 同名函数以及 `Model.forward` 所需 helper，不保留 tile2asc 的 case 构造接口。


## 3. 接口规范

### 算子原型

```python
paged_attention_kv_cache(Tensor q, Tensor k_cache, Tensor v_cache, Tensor cache_seqlens, Tensor page_table, bool causal=True) -> Tensor output
```

### 输入参数说明

| 参数 | 类型 | 支持 dtype | Shape | 描述 |
|---|---|---|---|---|
| `q` | Tensor | `bfloat16, float16` | q [B, Sq, Hq, D], k/v cache [Blocks, Page, Hkv, D], page_table [B, MaxBlocks]. | q input tensor |
| `k_cache` | Tensor | `bfloat16, float16` | q [B, Sq, Hq, D], k/v cache [Blocks, Page, Hkv, D], page_table [B, MaxBlocks]. | k_cache input tensor |
| `v_cache` | Tensor | `bfloat16, float16` | q [B, Sq, Hq, D], k/v cache [Blocks, Page, Hkv, D], page_table [B, MaxBlocks]. | v_cache input tensor |
| `cache_seqlens` | Tensor | `int32` | q [B, Sq, Hq, D], k/v cache [Blocks, Page, Hkv, D], page_table [B, MaxBlocks]. | cache_seqlens input tensor |
| `page_table` | Tensor | `int32` | q [B, Sq, Hq, D], k/v cache [Blocks, Page, Hkv, D], page_table [B, MaxBlocks]. | page_table input tensor |

### 属性

| 参数 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| `causal` | bool | `True` | causal operator attribute |

### 输出

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `output` | 由参考 forward 和输入 shape 决定 | 跟随参考实现输出 | golden 输出张量 |

### 规则与约束

- q shape 为 [B, Sq, Hq, D]，k_cache/v_cache shape 为 [Blocks, Page, Hkv, D]。
- Hq 必须能被 Hkv 整除；page_table/cache_seqlens 必须能重建有效 KV 序列。
- causal 控制 decoder causal mask；page_block_size 由 cache tensor 的 Page 维体现。

### 支持范围

| 维度 / 参数 | staging cases 覆盖 |
|---|---|
| `batch_size` | 1 ~ 8 |
| `cache_seqlen` | 64 ~ 512 |
| `causal` | {False, True} |
| `dtype` | {bfloat16, float16} |
| `headdim` | 64 ~ 128 |
| `max_blocks_per_seq` | 4 ~ 64 |
| `nheads_kv` | 4 ~ 16 |
| `nheads_q` | 8 ~ 32 |
| `num_blocks` | 16 ~ 512 |
| `page_block_size` | 16 ~ 16 |
| `seqlen_q` | 1 ~ 200 |
| `input dtype tuple` | bfloat16/bfloat16/bfloat16/int32/int32 ; float16/float16/float16/int32/int32 |

## 4. 精度要求

采用生态算子精度标准进行验证，主要使用平均相对误差（MERE）和最大相对误差（MARE）。

```text
MERE = avg(abs(actual - golden) / (abs(golden) + 1e-7))
MARE = max(abs(actual - golden) / (abs(golden) + 1e-7))
```

当 MERE < dtype 对应 Threshold 且 MARE < 10 * Threshold 时判定为通过。对 sparse/page/mask/TND 等结构化输入，比较前必须保证索引、page table、mask 或 prefix length 语义合法。

## 5. 标准 Golden 代码

完整 PyTorch 参考实现位于同目录 `golden.py`。正式入口为 schema 同名函数：

```python
def paged_attention_kv_cache(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, cache_seqlens: torch.Tensor, page_table: torch.Tensor, causal: bool = True) -> torch.Tensor:
```

该函数内部仅按输入 tensor shape 和正式 attrs 初始化参考 `Model`，然后调用 upstream `Model.forward`。

## 6. 额外信息

- `cases.yaml` / `cases.csv`：20 条 staging case，attrs 使用 YAML 原生类型。
- `baseline_perf_us` 和 `t_hw_us` 当前保持 `null`，需后续硬件补测。
- 本任务仍处于 `bench_lab/cv_agent_bench` staging 目录，正式并入 `tasks/level4` 前应再次跑性能与 parser 兼容性检查。
