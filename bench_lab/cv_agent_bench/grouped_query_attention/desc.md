# GroupedQueryAttention 算子 API 描述

## 1. 算子简介

标准 Grouped Query Attention，多个 query head 共享一组 KV head，通过 repeat KV head 后执行 scaled dot-product attention。

**主要应用场景**：

- Decoder-only LLM 的 GQA 注意力层
- KV cache 压缩后的推理 attention
- query head 多于 key/value head 的训练或推理场景

**算子特征**：

- q 使用 n_heads，k/v 使用 n_kv_heads
- K/V 按 group 重复到 query head 数
- 输出为 [B, n_heads, S, D]
- 难度等级：L4（FusedComposite）

## 2. 算子定义

### 数学公式

```text
K_rep = repeat_interleave(K, Hq / Hkv); V_rep = repeat_interleave(V, Hq / Hkv); Y = softmax(Q @ K_rep^T / sqrt(D)) @ V_rep
```

### Golden 语义来源

- 参考实现：`/root/asc/cv_agent_all/cv_agent/tile2asc/grouped_query_attention/model.py`
- staging golden 只保留 schema 同名函数以及 `Model.forward` 所需 helper，不保留 tile2asc 的 case 构造接口。


## 3. 接口规范

### 算子原型

```python
grouped_query_attention(Tensor q, Tensor k, Tensor v) -> Tensor output
```

### 输入参数说明

| 参数 | 类型 | 支持 dtype | Shape | 描述 |
|---|---|---|---|---|
| `q` | Tensor | `float16` | q [B, Hq, S, D], k/v [B, Hkv, S, D], Hq % Hkv == 0. | q input tensor |
| `k` | Tensor | `float16` | q [B, Hq, S, D], k/v [B, Hkv, S, D], Hq % Hkv == 0. | k input tensor |
| `v` | Tensor | `float16` | q [B, Hq, S, D], k/v [B, Hkv, S, D], Hq % Hkv == 0. | v input tensor |

### 属性

| 参数 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| - | - | - | 无正式算子属性；shape/dtype 仅由 case 输入张量决定 |

### 输出

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `output` | 由参考 forward 和输入 shape 决定 | 跟随参考实现输出 | golden 输出张量 |

### 规则与约束

- q shape 为 [B, Hq, S, D]，k/v shape 为 [B, Hkv, S, D]。
- Hq 必须能被 Hkv 整除，K/V 在 head 维 repeat 到 Hq 后参与 attention。
- 输出 shape 与 q 一致。

### 支持范围

| 维度 / 参数 | staging cases 覆盖 |
|---|---|
| `batch_size` | 1 ~ 16 |
| `d_k` | 128 ~ 512 |
| `dtype` | {float16} |
| `n_heads` | 4 ~ 32 |
| `n_kv_heads` | 1 ~ 8 |
| `seq_len` | 128 ~ 2048 |
| `input dtype tuple` | float16/float16/float16 |

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
def grouped_query_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
```

该函数内部仅按输入 tensor shape 和正式 attrs 初始化参考 `Model`，然后调用 upstream `Model.forward`。

## 6. 额外信息

- `cases.yaml` / `cases.csv`：20 条 staging case，attrs 使用 YAML 原生类型。
- `baseline_perf_us` 和 `t_hw_us` 当前保持 `null`，需后续硬件补测。
- 本任务仍处于 `bench_lab/cv_agent_bench` staging 目录，正式并入 `tasks/level4` 前应再次跑性能与 parser 兼容性检查。
