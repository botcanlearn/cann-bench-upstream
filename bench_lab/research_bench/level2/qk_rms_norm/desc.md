# QkRmsNorm 算子 API 描述

## 1. 算子简介

`QkRmsNorm` 面向 Attention 前处理中的 Q/K norm 模型变体，对 query 和 key 两路张量分别执行 RMSNorm。该算子用于覆盖热门 LLM / VLM 中常见的 QK norm 结构，目标是减少两次独立 RMSNorm 带来的重复访存。

**算子特征**：

- 难度等级：L2（FusedComposite）
- 输入为 `q`、`k`、`gamma_q`、`gamma_k`
- 输出为 `(q_out, k_out)`
- Q/K 各自沿最后一维归一化，`gamma_q` 与 `gamma_k` 独立

## 2. 算子定义

对任意输入 `x`，RMSNorm 定义为：

```text
rms = sqrt(mean(x^2, dim=-1, keepdim=True) + eps)
y = x / rms * gamma
```

`QkRmsNorm` 分别计算：

```text
q_out = rms_norm(q, gamma_q, eps)
k_out = rms_norm(k, gamma_k, eps)
```

## 3. 接口规范

```python
cann_bench.qk_rms_norm(
    Tensor q,
    Tensor k,
    Tensor gamma_q,
    Tensor gamma_k,
    float eps=1e-6,
) -> (Tensor q_out, Tensor k_out)
```

### 输入参数

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `q` | `[..., D]` | float16 / bfloat16 / float32 | Query 张量 |
| `k` | `[..., D]` | float16 / bfloat16 / float32 | Key 张量 |
| `gamma_q` | `[D]` | 与 `q` 相同 | Query 缩放参数 |
| `gamma_k` | `[D]` | 与 `k` 相同 | Key 缩放参数 |
| `eps` | 标量 | float | 数值稳定项，必须大于 0 |

### 输出

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `q_out` | 与 `q` 相同 | 与 `q` 相同 | Query RMSNorm 输出 |
| `k_out` | 与 `k` 相同 | 与 `k` 相同 | Key RMSNorm 输出 |

### 数据类型

| 输入 dtype | 输出 dtype |
|---|---|
| float16 | float16 |
| bfloat16 | bfloat16 |
| float32 | float32 |

### 规则与约束

- `q.shape[-1] == gamma_q.shape[0]`
- `k.shape[-1] == gamma_k.shape[0]`
- `q` 与 `k` 的前导维度可不同，但最后一维必须分别匹配对应 gamma。
- `eps > 0`；首版不把 `NaN`、`inf` 或负数 `eps` 作为有效输入。
- 参考实现使用 float32 中间计算，再 cast 回输入 dtype。

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| rank(`q`), rank(`k`) | 2 ~ 4 | 覆盖 `[B, S, D]` 与 `[B, S, H, D]` |
| `D` | 16 ~ 4096 | 公开 case 覆盖 16 / 64 / 128 / 256 / 512 / 1024 |
| `eps` | `1e-12` ~ `1e-3` | 必须为正数 |

## 4. 精度要求

采用[生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)进行验证。

**误差指标**：

1. 平均相对误差（MERE）：采样点中相对误差平均值

   $$
   \text{MERE} = \text{avg}(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

2. 最大相对误差（MARE）：采样点中相对误差最大值

   $$
   \text{MARE} = \max(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

**通过标准**：

| 数据类型 | FLOAT16 | BFLOAT16 | FLOAT32 | HiFLOAT32 | FLOAT8 E4M3 | FLOAT8 E5M2 |
|----------|---------|----------|---------|-----------|-------------|-------------|
| **通过阈值(Threshold)** | 2^-10 | 2^-7 | 2^-13 | 2^-11 | 2^-3 | 2^-2 |

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过。

## 5. 标准 Golden 代码

```python
#!/usr/bin/python3
# coding=utf-8

import torch


def _rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    out_dtype = x.dtype
    x_f = x.float()
    gamma_f = gamma.float()
    variance = x_f.pow(2).mean(dim=-1, keepdim=True)
    y = x_f * torch.rsqrt(variance + eps)
    y = y * gamma_f
    return y.to(out_dtype)


def qk_rms_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    gamma_q: torch.Tensor,
    gamma_k: torch.Tensor,
    eps: float = 1e-6,
):
    """Q/K 双路 RMSNorm reference."""
    return _rms_norm(q, gamma_q, eps), _rms_norm(k, gamma_k, eps)
```
