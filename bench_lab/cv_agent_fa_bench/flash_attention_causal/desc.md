# FlashAttentionCausal 算子 API 描述

## 1. 算子简介

带 causal mask 的 scaled dot-product attention。目录名沿用源算子拼写 casual，语义为 causal。

**主要应用场景**：

- 自回归语言模型训练
- decoder causal attention
- 不能访问未来 token 的序列建模

**算子特征**：

- 内部构造下三角 causal mask
- mask 上三角位置填充为 -inf
- 输出 shape 与 q 一致
- 难度等级：L4（FusedComposite）

## 2. 算子定义

### 数学公式

```text
Y = softmax((Q @ K^T / sqrt(D)) + causal_mask) @ V
```

### Golden 语义来源

- 参考实现：`/root/asc/cv_agent_all/cv_agent/tile2asc/flash_attention_casual/model.py`
- staging golden 只保留 schema 同名函数以及 `Model.forward` 所需 helper，不保留 tile2asc 的 case 构造接口。

- `flash_attention_casual` 在 staging 中规范化命名为 `flash_attention_causal`，语义仍来自原始目录。

## 3. 接口规范

### 算子原型

```python
flash_attention_causal(Tensor q, Tensor k, Tensor v) -> Tensor output
```

### 输入参数说明

| 参数 | 类型 | 支持 dtype | Shape | 描述 |
|---|---|---|---|---|
| `q` | Tensor | `bfloat16, float16` | q/k/v [B, H, S, D] with causal lower-triangular masking. | q input tensor |
| `k` | Tensor | `bfloat16, float16` | q/k/v [B, H, S, D] with causal lower-triangular masking. | k input tensor |
| `v` | Tensor | `bfloat16, float16` | q/k/v [B, H, S, D] with causal lower-triangular masking. | v input tensor |

### 属性

| 参数 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| - | - | - | 无正式算子属性；shape/dtype 仅由 case 输入张量决定 |

### 输出

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `output` | 由参考 forward 和输入 shape 决定 | 跟随参考实现输出 | golden 输出张量 |

### 规则与约束

- q/k/v shape 必须一致，均为 [B, H, S, D]。
- forward 内部构造下三角 causal mask，仅允许当前位置访问自身及历史 token。
- 输出 shape 与 q 一致。

### 支持范围

| 维度 / 参数 | staging cases 覆盖 |
|---|---|
| `batch_size` | 1 ~ 8 |
| `dtype` | {bfloat16, float16} |
| `head_dim` | 128 ~ 512 |
| `num_heads` | 1 ~ 32 |
| `seq_len` | 1 ~ 4096 |
| `input dtype tuple` | bfloat16/bfloat16/bfloat16 ; float16/float16/float16 |

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
def flash_attention_causal(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
```

该函数内部仅按输入 tensor shape 和正式 attrs 初始化参考 `Model`，然后调用 upstream `Model.forward`。

## 6. 额外信息

- `cases.yaml` / `cases.csv`：20 条 staging case，attrs 使用 YAML 原生类型。
- `baseline_perf_us` 和 `t_hw_us` 当前保持 `null`，需后续硬件补测。
- 本任务仍处于 `bench_lab/cv_agent_bench` staging 目录，正式并入 `tasks/level4` 前应再次跑性能与 parser 兼容性检查。
