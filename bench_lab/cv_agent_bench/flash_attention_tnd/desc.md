# FlashAttentionTnd 算子 API 描述

## 1. 算子简介

TND 变长序列 attention，多个 batch 的 token 在 T 维拼接，actual_q_len/actual_kv_len 提供每段前缀和边界。

**主要应用场景**：

- 变长 batch 的 packed attention
- 避免 padding 的 TND 布局
- prefill 多请求拼接执行

**算子特征**：

- q shape 为 [Tq, H, D]
- k/v shape 为 [Tkv, H, D]
- actual_q_len 与 actual_kv_len 为 batch 前缀和
- 难度等级：L4（FusedComposite）

## 2. 算子定义

### 数学公式

```text
Split packed TND inputs by actual length prefixes and compute attention per segment.
```

### Golden 语义来源

- 参考实现：`/root/asc/cv_agent_all/cv_agent/tile2asc/flash_attention_tnd/model.py`
- staging golden 只保留 schema 同名函数以及 `Model.forward` 所需 helper，不保留 tile2asc 的 case 构造接口。


## 3. 接口规范

### 算子原型

```python
flash_attention_tnd(Tensor q, Tensor k, Tensor v, Tensor actual_q_len, Tensor actual_kv_len) -> Tensor output
```

### 输入参数说明

| 参数 | 类型 | 支持 dtype | Shape | 描述 |
|---|---|---|---|---|
| `q` | Tensor | `bfloat16, float16` | packed q [sum(Sq_i), H, D], k/v [sum(Skv_i), H, D], prefix length tensors [B]. | q input tensor |
| `k` | Tensor | `bfloat16, float16` | packed q [sum(Sq_i), H, D], k/v [sum(Skv_i), H, D], prefix length tensors [B]. | k input tensor |
| `v` | Tensor | `bfloat16, float16` | packed q [sum(Sq_i), H, D], k/v [sum(Skv_i), H, D], prefix length tensors [B]. | v input tensor |
| `actual_q_len` | Tensor | `int64` | packed q [sum(Sq_i), H, D], k/v [sum(Skv_i), H, D], prefix length tensors [B]. | actual_q_len input tensor |
| `actual_kv_len` | Tensor | `int64` | packed q [sum(Sq_i), H, D], k/v [sum(Skv_i), H, D], prefix length tensors [B]. | actual_kv_len input tensor |

### 属性

| 参数 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| - | - | - | 无正式算子属性；shape/dtype 仅由 case 输入张量决定 |

### 输出

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `output` | 由参考 forward 和输入 shape 决定 | 跟随参考实现输出 | golden 输出张量 |

### 规则与约束

- q shape 为 [sum(Sq_i), H, D]，k/v shape 为 [sum(Skv_i), H, D]。
- actual_q_len 和 actual_kv_len 为单调递增 prefix length tensor。
- actual_q_len[-1] 必须等于 packed q 的第一维，actual_kv_len[-1] 必须等于 packed k/v 的第一维。

### 支持范围

| 维度 / 参数 | staging cases 覆盖 |
|---|---|
| `dim` | 64 ~ 256 |
| `dtype` | {bfloat16, float16} |
| `kv_len_prefix` | [128, 256] / [64] / [512] / [64, 128, 192, 256] / [128, 256] / ... |
| `n_heads` | 4 ~ 16 |
| `q_len_prefix` | [128, 256] / [64] / [512] / [64, 128, 192, 256] / [64, 128] / ... |
| `input dtype tuple` | bfloat16/bfloat16/bfloat16/int64/int64 ; float16/float16/float16/int64/int64 |

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
def flash_attention_tnd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, actual_q_len: torch.Tensor, actual_kv_len: torch.Tensor) -> torch.Tensor:
```

该函数内部仅按输入 tensor shape 和正式 attrs 初始化参考 `Model`，然后调用 upstream `Model.forward`。

## 6. 额外信息

- `cases.yaml` / `cases.csv`：20 条 staging case，attrs 使用 YAML 原生类型。
- `baseline_perf_us` 和 `t_hw_us` 当前保持 `null`，需后续硬件补测。
- 本任务仍处于 `bench_lab/cv_agent_bench` staging 目录，正式并入 `tasks/level4` 前应再次跑性能与 parser 兼容性检查。
