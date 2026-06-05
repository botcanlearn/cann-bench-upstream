# DenseSparseAttention 算子 API 描述

## 1. 算子简介

Dense-Sparse Attention，query 从 paged/flatten KV cache 中按 indices 选择 topk token 计算注意力。

**主要应用场景**：

- FlashMLA 风格稀疏 KV 选择
- 长上下文 topk 检索 attention
- 压缩 KV cache 推理

**算子特征**：

- kv_cache shape 为 [num_blocks,page_block_size,1,headdim_qk]
- indices shape 为 [B,Sq,topk]
- 输出最后一维为 headdim_v
- 难度等级：L4（FusedComposite）

## 2. 算子定义

### 数学公式

```text
Gather sparse KV entries from paged cache by indices, then compute top-k attention per query/head.
```

### Golden 语义来源

- 参考实现：`/root/asc/cv_agent_all/cv_agent/tile2asc/dense_sparse_attention/model.py`
- staging golden 只保留 schema 同名函数以及 `Model.forward` 所需 helper，不保留 tile2asc 的 case 构造接口。


## 3. 接口规范

### 算子原型

```python
dense_sparse_attention(Tensor q, Tensor kv_cache, Tensor indices) -> Tensor output
```

### 输入参数说明

| 参数 | 类型 | 支持 dtype | Shape | 描述 |
|---|---|---|---|---|
| `q` | Tensor | `bfloat16` | q [B, Sq, H, Dqk], kv_cache [Blocks, Page, 1, Dqk], indices [B, Sq, H, TopK]. | q input tensor |
| `kv_cache` | Tensor | `bfloat16` | q [B, Sq, H, Dqk], kv_cache [Blocks, Page, 1, Dqk], indices [B, Sq, H, TopK]. | kv_cache input tensor |
| `indices` | Tensor | `int32` | q [B, Sq, H, Dqk], kv_cache [Blocks, Page, 1, Dqk], indices [B, Sq, H, TopK]. | indices input tensor |

### 属性

| 参数 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| - | - | - | 无正式算子属性；shape/dtype 仅由 case 输入张量决定 |

### 输出

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `output` | 由参考 forward 和输入 shape 决定 | 跟随参考实现输出 | golden 输出张量 |

### 规则与约束

- q shape 为 [B, Sq, H, Dqk]，kv_cache shape 为 [Blocks, Page, 1, Dqk]。
- indices shape 为 [B, Sq, H, TopK]；forward 会将 indices clamp 到 flatten KV cache 范围内。
- headdim_v 由参考模型按输入 Dqk 推导，要求 headdim_v <= Dqk。

### 支持范围

| 维度 / 参数 | staging cases 覆盖 |
|---|---|
| `batch_size` | 1 ~ 16 |
| `headdim_qk` | 128 ~ 576 |
| `headdim_v` | 128 ~ 512 |
| `nheads` | 4 ~ 16 |
| `num_blocks` | 64 ~ 512 |
| `page_block_size` | 16 ~ 16 |
| `seqlen_q` | 1 ~ 256 |
| `topk` | 8 ~ 128 |
| `input dtype tuple` | bfloat16/bfloat16/int32 |

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
def dense_sparse_attention(q: torch.Tensor, kv_cache: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
```

该函数内部仅按输入 tensor shape 和正式 attrs 初始化参考 `Model`，然后调用 upstream `Model.forward`。

## 6. 额外信息

- `cases.yaml` / `cases.csv`：20 条 staging case，attrs 使用 YAML 原生类型。
- `baseline_perf_us` 和 `t_hw_us` 当前保持 `null`，需后续硬件补测。
- 本任务仍处于 `bench_lab/cv_agent_bench` staging 目录，正式并入 `tasks/level4` 前应再次跑性能与 parser 兼容性检查。
