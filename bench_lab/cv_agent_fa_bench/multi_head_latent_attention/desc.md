# MultiHeadLatentAttention 算子 API 描述

## 1. 算子简介

Paged 单 KV head latent attention，KV cache 存储已生成的单头 latent 表示，并通过 block_table 从 paged cache 还原。本 golden 不包含上游投影矩阵。

**主要应用场景**：

- paged 单 KV head cache 推理
- latent KV cache attention
- headdim_qk 与 headdim_v 不同的 attention

**算子特征**：

- q shape 为 [B,Sq,Hq,headdim_qk]
- kv_cache 使用 [num_blocks,page_block_size,1,headdim_qk]
- V 使用 KV 的前 headdim_v 维
- 难度等级：L4（FusedComposite）

## 2. 算子定义

### 数学公式

```text
Reconstruct latent KV cache through block_table and compute broadcast single-KV-head attention.
```

### Golden 语义来源

- 参考实现：`/root/asc/cv_agent_all/cv_agent/tile2asc/multi_head_latent_attention/model.py`
- staging golden 只保留 schema 同名函数以及 `Model.forward` 所需 helper，不保留 tile2asc 的 case 构造接口。


## 3. 接口规范

### 算子原型

```python
multi_head_latent_attention(Tensor q, Tensor kv_cache, Tensor block_table, Tensor cache_seqlens, bool causal=True) -> Tensor output
```

### 输入参数说明

| 参数 | 类型 | 支持 dtype | Shape | 描述 |
|---|---|---|---|---|
| `q` | Tensor | `bfloat16` | q [B, Sq, Hq, Dqk], latent kv_cache [Blocks, Page, 1, Dqk], block_table [B, MaxBlocks]. | q input tensor |
| `kv_cache` | Tensor | `bfloat16` | q [B, Sq, Hq, Dqk], latent kv_cache [Blocks, Page, 1, Dqk], block_table [B, MaxBlocks]. | kv_cache input tensor |
| `block_table` | Tensor | `int32` | q [B, Sq, Hq, Dqk], latent kv_cache [Blocks, Page, 1, Dqk], block_table [B, MaxBlocks]. | block_table input tensor |
| `cache_seqlens` | Tensor | `int32` | q [B, Sq, Hq, Dqk], latent kv_cache [Blocks, Page, 1, Dqk], block_table [B, MaxBlocks]. | cache_seqlens input tensor |

### 属性

| 参数 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| `causal` | bool | `True` | causal operator attribute |

### 输出

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `output` | 由参考 forward 和输入 shape 决定 | 跟随参考实现输出 | golden 输出张量 |

### 规则与约束

- q shape 为 [B, Sq, Hq, Dqk]，kv_cache shape 为 [Blocks, Page, 1, Dqk]。
- block_table/cache_seqlens 必须能重建每个 batch 的有效 KV 序列。
- causal 控制 decoder causal mask；page_block_size 由 kv_cache 的 Page 维体现。

### 支持范围

| 维度 / 参数 | staging cases 覆盖 |
|---|---|
| `batch_size` | 1 ~ 8 |
| `cache_seqlen` | 64 ~ 1024 |
| `causal` | {False, True} |
| `dtype` | {bfloat16} |
| `headdim_qk` | 576 ~ 576 |
| `headdim_v` | 512 ~ 512 |
| `max_blocks_per_seq` | 4 ~ 64 |
| `nheads_q` | 16 ~ 16 |
| `num_blocks` | 4 ~ 512 |
| `page_block_size` | 16 ~ 16 |
| `seqlen_q` | 1 ~ 128 |
| `input dtype tuple` | bfloat16/bfloat16/int32/int32 |

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
def multi_head_latent_attention(q: torch.Tensor, kv_cache: torch.Tensor, block_table: torch.Tensor, cache_seqlens: torch.Tensor, causal: bool = True) -> torch.Tensor:
```

该函数内部仅按输入 tensor shape 和正式 attrs 初始化参考 `Model`，然后调用 upstream `Model.forward`。

## 6. 额外信息

- `cases.yaml` / `cases.csv`：20 条 staging case，attrs 使用 YAML 原生类型。
- `baseline_perf_us` 和 `t_hw_us` 当前保持 `null`，需后续硬件补测。
- 本任务仍处于 `bench_lab/cv_agent_bench` staging 目录，正式并入 `tasks/level4` 前应再次跑性能与 parser 兼容性检查。
