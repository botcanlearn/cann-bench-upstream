# FlashAttentionSink 算子 API 描述

## 1. 算子简介

Sink token attention，将 sink_k/sink_v 拼接到普通 K/V 前参与 softmax，使 query 始终可以关注固定 sink token。

**主要应用场景**：

- attention sink 推理优化
- 长上下文中固定全局 token 保留
- 带 sink token 的 decoder attention

**算子特征**：

- 额外输入 sink_k 与 sink_v
- KV 序列长度变为 sink_size + seq_len_kv
- 输出 shape 与 q 一致
- 难度等级：L4（FusedComposite）

## 2. 算子定义

### 数学公式

```text
Concatenate sink K/V with ordinary K/V, then compute scaled dot-product attention.
```

### Golden 语义来源

- 参考实现：`/root/asc/cv_agent_all/cv_agent/tile2asc/flash_attention_sink/model.py`
- staging golden 只保留 schema 同名函数以及 `Model.forward` 所需 helper，不保留 tile2asc 的 case 构造接口。


## 3. 接口规范

### 算子原型

```python
flash_attention_sink(Tensor q, Tensor k, Tensor v, Tensor sink_k, Tensor sink_v) -> Tensor output
```

### 输入参数说明

| 参数 | 类型 | 支持 dtype | Shape | 描述 |
|---|---|---|---|---|
| `q` | Tensor | `bfloat16, float16` | q [B, H, Sq, D], k/v [B, H, Skv, D], sink_k/sink_v [B, H, Sink, D]. | q input tensor |
| `k` | Tensor | `bfloat16, float16` | q [B, H, Sq, D], k/v [B, H, Skv, D], sink_k/sink_v [B, H, Sink, D]. | k input tensor |
| `v` | Tensor | `bfloat16, float16` | q [B, H, Sq, D], k/v [B, H, Skv, D], sink_k/sink_v [B, H, Sink, D]. | v input tensor |
| `sink_k` | Tensor | `bfloat16, float16` | q [B, H, Sq, D], k/v [B, H, Skv, D], sink_k/sink_v [B, H, Sink, D]. | sink_k input tensor |
| `sink_v` | Tensor | `bfloat16, float16` | q [B, H, Sq, D], k/v [B, H, Skv, D], sink_k/sink_v [B, H, Sink, D]. | sink_v input tensor |

### 属性

| 参数 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| - | - | - | 无正式算子属性；shape/dtype 仅由 case 输入张量决定 |

### 输出

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `output` | 由参考 forward 和输入 shape 决定 | 跟随参考实现输出 | golden 输出张量 |

### 规则与约束

- q shape 为 [B, H, Sq, D]，k/v shape 为 [B, H, Skv, D]。
- sink_k/sink_v shape 为 [B, H, Sink, D]，batch、head、head_dim 必须与 q/k/v 一致。
- sink_size 由 sink tensor shape 体现，不作为正式 schema attr。

### 支持范围

| 维度 / 参数 | staging cases 覆盖 |
|---|---|
| `batch_size` | 1 ~ 8 |
| `d_k` | 128 ~ 512 |
| `dtype` | {bfloat16, float16} |
| `n_heads` | 4 ~ 32 |
| `seq_len_kv` | 64 ~ 4096 |
| `seq_len_q` | 1 ~ 200 |
| `sink_size` | 16 ~ 128 |
| `input dtype tuple` | bfloat16/bfloat16/bfloat16/bfloat16/bfloat16 ; float16/float16/float16/float16/float16 |

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
def flash_attention_sink(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sink_k: torch.Tensor, sink_v: torch.Tensor) -> torch.Tensor:
```

该函数内部仅按输入 tensor shape 和正式 attrs 初始化参考 `Model`，然后调用 upstream `Model.forward`。

## 6. 额外信息

- `cases.yaml` / `cases.csv`：20 条 staging case，attrs 使用 YAML 原生类型。
- `baseline_perf_us` 和 `t_hw_us` 当前保持 `null`，需后续硬件补测。
- 本任务仍处于 `bench_lab/cv_agent_bench` staging 目录，正式并入 `tasks/level4` 前应再次跑性能与 parser 兼容性检查。
