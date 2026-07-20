# FlashAttentionPa 算子 API 描述

## 1. 算子简介

Paged Attention 版本，将 paged KV cache 按 block_table 还原为连续 K/V 后执行 attention。

**主要应用场景**：

- LLM paged KV cache 推理
- 按 block 管理长上下文 KV 的 decoding
- 验证 block_table 映射正确性

**算子特征**：

- k_cache/v_cache 使用 [block_num, block_size, H, D]
- block_table 描述每个 batch 的逻辑块到物理块映射
- 还原连续 K/V 后计算标准 attention
- 难度等级：L4（FusedComposite）

## 2. 算子定义

### 数学公式

```text
Gather paged K/V cache by block_table, then compute scaled dot-product attention against q.
```

### Golden 语义来源

- 参考实现：`/root/asc/cv_agent_all/cv_agent/tile2asc/flash_attention_pa/model.py`
- staging golden 只保留 schema 同名函数以及 `Model.forward` 所需 helper，不保留 tile2asc 的 case 构造接口。


## 3. 接口规范

### 算子原型

```python
flash_attention_pa(Tensor q, Tensor k_cache, Tensor v_cache, Tensor block_table) -> Tensor output
```

### 输入参数说明

| 参数 | 类型 | 支持 dtype | Shape | 描述 |
|---|---|---|---|---|
| `q` | Tensor | `float16` | q [B, H, S, D], paged k/v cache [Blocks, Block, H, D], block_table [B, Pages]. | q input tensor |
| `k_cache` | Tensor | `float16` | q [B, H, S, D], paged k/v cache [Blocks, Block, H, D], block_table [B, Pages]. | k_cache input tensor |
| `v_cache` | Tensor | `float16` | q [B, H, S, D], paged k/v cache [Blocks, Block, H, D], block_table [B, Pages]. | v_cache input tensor |
| `block_table` | Tensor | `int32` | q [B, H, S, D], paged k/v cache [Blocks, Block, H, D], block_table [B, Pages]. | block_table input tensor |

### 属性

| 参数 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| - | - | - | 无正式算子属性；shape/dtype 仅由 case 输入张量决定 |

### 输出

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `output` | 由参考 forward 和输入 shape 决定 | 跟随参考实现输出 | golden 输出张量 |

### 规则与约束

- q shape 为 [B, H, S, D]，k_cache/v_cache shape 为 [Blocks, Block, H, D]。
- block_table shape 为 [B, Pages]，其中索引必须落在 cache block 范围内。
- block_size 由 cache tensor shape 体现，不作为正式 schema attr。

### 支持范围

| 维度 / 参数 | staging cases 覆盖 |
|---|---|
| `batch_size` | 1 ~ 16 |
| `block_size` | 64 ~ 256 |
| `dtype` | {float16} |
| `head_dim` | 128 ~ 512 |
| `n_heads` | 4 ~ 32 |
| `seq_len` | 1 ~ 8192 |
| `input dtype tuple` | float16/float16/float16/int32 |

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
def flash_attention_pa(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, block_table: torch.Tensor) -> torch.Tensor:
```

该函数内部仅按输入 tensor shape 和正式 attrs 初始化参考 `Model`，然后调用 upstream `Model.forward`。

## 6. 额外信息

- `cases.yaml` / `cases.csv`：20 条 staging case，attrs 使用 YAML 原生类型。
- `baseline_perf_us` 和 `t_hw_us` 当前保持 `null`，需后续硬件补测。
- 本任务仍处于 `bench_lab/cv_agent_bench` staging 目录，正式并入 `tasks/level4` 前应再次跑性能与 parser 兼容性检查。
