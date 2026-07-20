# FlashAttentionSinkPa 算子 API 描述

## 1. 算子简介

Paged Attention 与 sink token 结合的推理 attention，先根据 block_table/context_lens 还原 paged KV，再拼接 sink KV。

**主要应用场景**：

- 带 attention sink 的 paged KV cache decoding
- 长上下文推理
- 固定 sink token 与动态上下文混合

**算子特征**：

- 输入包含 k_cache/v_cache、block_table、context_lens、sink_k/sink_v
- 每个 batch 按 context_lens 截断有效 KV
- 输出为单步或短序列 query 的 attention 结果
- 难度等级：L4（FusedComposite）

## 2. 算子定义

### 数学公式

```text
Reconstruct paged cache per batch, prepend sink tokens, then compute attention for the single query token.
```

### Golden 语义来源

- 参考实现：`/root/asc/cv_agent_all/cv_agent/tile2asc/flash_attention_sink_pa/model.py`
- staging golden 只保留 schema 同名函数以及 `Model.forward` 所需 helper，不保留 tile2asc 的 case 构造接口。


## 3. 接口规范

### 算子原型

```python
flash_attention_sink_pa(Tensor q, Tensor k_cache, Tensor v_cache, Tensor block_table, Tensor context_lens, Tensor sink_k, Tensor sink_v) -> Tensor output
```

### 输入参数说明

| 参数 | 类型 | 支持 dtype | Shape | 描述 |
|---|---|---|---|---|
| `q` | Tensor | `bfloat16, float16` | q [B, H, 1, D], paged cache [TotalBlocks, H, Block, D], sink tensors [B, H, Sink, D]. | q input tensor |
| `k_cache` | Tensor | `bfloat16, float16` | q [B, H, 1, D], paged cache [TotalBlocks, H, Block, D], sink tensors [B, H, Sink, D]. | k_cache input tensor |
| `v_cache` | Tensor | `bfloat16, float16` | q [B, H, 1, D], paged cache [TotalBlocks, H, Block, D], sink tensors [B, H, Sink, D]. | v_cache input tensor |
| `block_table` | Tensor | `int32` | q [B, H, 1, D], paged cache [TotalBlocks, H, Block, D], sink tensors [B, H, Sink, D]. | block_table input tensor |
| `context_lens` | Tensor | `int32` | q [B, H, 1, D], paged cache [TotalBlocks, H, Block, D], sink tensors [B, H, Sink, D]. | context_lens input tensor |
| `sink_k` | Tensor | `bfloat16, float16` | q [B, H, 1, D], paged cache [TotalBlocks, H, Block, D], sink tensors [B, H, Sink, D]. | sink_k input tensor |
| `sink_v` | Tensor | `bfloat16, float16` | q [B, H, 1, D], paged cache [TotalBlocks, H, Block, D], sink tensors [B, H, Sink, D]. | sink_v input tensor |

### 属性

| 参数 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| - | - | - | 无正式算子属性；shape/dtype 仅由 case 输入张量决定 |

### 输出

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `output` | 由参考 forward 和输入 shape 决定 | 跟随参考实现输出 | golden 输出张量 |

### 规则与约束

- q shape 为 [B, H, 1, D]，k_cache/v_cache shape 为 [TotalBlocks, H, Block, D]。
- block_table 和 context_lens 必须能覆盖每个 batch 的 paged cache 上下文。
- sink_k/sink_v shape 为 [B, H, Sink, D]；sink_size/block_size 由输入 tensor shape 体现。

### 支持范围

| 维度 / 参数 | staging cases 覆盖 |
|---|---|
| `batch_size` | 1 ~ 16 |
| `block_size` | 64 ~ 64 |
| `d_k` | 128 ~ 512 |
| `dtype` | {bfloat16, float16} |
| `n_heads` | 8 ~ 32 |
| `seq_len_kv` | 1 ~ 4096 |
| `sink_size` | 64 ~ 128 |
| `input dtype tuple` | bfloat16/bfloat16/bfloat16/int32/int32/bfloat16/bfloat16 ; float16/float16/float16/int32/int32/float16/float16 |

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
def flash_attention_sink_pa(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, block_table: torch.Tensor, context_lens: torch.Tensor, sink_k: torch.Tensor, sink_v: torch.Tensor) -> torch.Tensor:
```

该函数内部仅按输入 tensor shape 和正式 attrs 初始化参考 `Model`，然后调用 upstream `Model.forward`。

## 6. 额外信息

- `cases.yaml` / `cases.csv`：20 条 staging case，attrs 使用 YAML 原生类型。
- `baseline_perf_us` 和 `t_hw_us` 当前保持 `null`，需后续硬件补测。
- 本任务仍处于 `bench_lab/cv_agent_bench` staging 目录，正式并入 `tasks/level4` 前应再次跑性能与 parser 兼容性检查。
