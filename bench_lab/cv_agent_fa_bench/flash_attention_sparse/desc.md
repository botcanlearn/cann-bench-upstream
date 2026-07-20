# FlashAttentionSparse 算子 API 描述

## 1. 算子简介

基于稀疏索引的 FlashAttention，每个 query 位置只从 indices 指定的 topk KV token 中计算注意力。

**主要应用场景**：

- 长序列稀疏 attention
- topk 检索式 KV 选择
- GQA 稀疏推理验证

**算子特征**：

- indices shape 为 [B, Sq, Hkv, topk]
- nheads_q 可为 nheads_kv 的整数倍
- 按 query head 映射到对应 KV head
- 难度等级：L4（FusedComposite）

## 2. 算子定义

### 数学公式

```text
Gather sparse KV entries by indices and compute sparse top-k attention.
```

### Golden 语义来源

- 参考实现：`/root/asc/cv_agent_all/cv_agent/tile2asc/flash_attention_sparse/model.py`
- staging golden 只保留 schema 同名函数以及 `Model.forward` 所需 helper，不保留 tile2asc 的 case 构造接口。


## 3. 接口规范

### 算子原型

```python
flash_attention_sparse(Tensor q, Tensor kv, Tensor indices) -> Tensor output
```

### 输入参数说明

| 参数 | 类型 | 支持 dtype | Shape | 描述 |
|---|---|---|---|---|
| `q` | Tensor | `bfloat16, float16` | q [B, Sq, Hq, D], kv [B, Skv, Hkv, D], indices [B, Sq, Hkv, TopK]. | q input tensor |
| `kv` | Tensor | `bfloat16, float16` | q [B, Sq, Hq, D], kv [B, Skv, Hkv, D], indices [B, Sq, Hkv, TopK]. | kv input tensor |
| `indices` | Tensor | `int32` | q [B, Sq, Hq, D], kv [B, Skv, Hkv, D], indices [B, Sq, Hkv, TopK]. | indices input tensor |

### 属性

| 参数 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| - | - | - | 无正式算子属性；shape/dtype 仅由 case 输入张量决定 |

### 输出

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `output` | 由参考 forward 和输入 shape 决定 | 跟随参考实现输出 | golden 输出张量 |

### 规则与约束

- q shape 为 [B, Sq, Hq, D]，kv shape 为 [B, Skv, Hkv, D]。
- Hq 必须能被 Hkv 整除；indices shape 为 [B, Sq, Hkv, TopK]。
- indices 会 clamp 到合法 KV 范围；topK 由 indices 最后一维体现，不作为正式 schema attr。

### 支持范围

| 维度 / 参数 | staging cases 覆盖 |
|---|---|
| `batch_size` | 1 ~ 4 |
| `dtype` | {bfloat16, float16} |
| `headdim` | 128 ~ 512 |
| `nheads_kv` | 1 ~ 8 |
| `nheads_q` | 8 ~ 32 |
| `seqlen_kv` | 128 ~ 4096 |
| `seqlen_q` | 1 ~ 1024 |
| `topk` | 8 ~ 256 |
| `input dtype tuple` | bfloat16/bfloat16/int32 ; float16/float16/int32 |

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
def flash_attention_sparse(q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
```

该函数内部仅按输入 tensor shape 和正式 attrs 初始化参考 `Model`，然后调用 upstream `Model.forward`。

## 6. 额外信息

- `cases.yaml` / `cases.csv`：20 条 staging case，attrs 使用 YAML 原生类型。
- `baseline_perf_us` 和 `t_hw_us` 当前保持 `null`，需后续硬件补测。
- 当前 golden 保留 upstream Python 循环实现，存在大 shape 性能风险；这是性能风险，不是语义错误。
- 本任务仍处于 `bench_lab/cv_agent_bench` staging 目录，正式并入 `tasks/level4` 前应再次跑性能与 parser 兼容性检查。
