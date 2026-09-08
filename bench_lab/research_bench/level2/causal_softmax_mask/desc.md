# CausalSoftmaxMask 算子 API 描述

## 1. 算子简介

`CausalSoftmaxMask` 覆盖 decoder attention 中常见的 score 后处理：scale、causal mask、padding/additive mask 与 softmax 融合。该算子用于减少 attention score 多次写回和 mask 多次读取。

**算子特征**：

- 难度等级：L2（FusedComposite）
- 输入为 attention score 和可选 mask
- 支持 causal / non-causal 两种模式
- 使用 float32 中间计算 softmax，输出 cast 回输入 dtype

## 2. 算子定义

给定 `scores`，先执行：

```text
x = scores * scale
```

若 `causal=True`，对最后两个维度 `[Sq, Skv]` 应用右对齐 causal mask。对 decode 场景 `Sq < Skv`，第 `i` 个 query token 可以访问 `j <= Skv - Sq + i` 的 key token。

若 `attention_mask` 非空：

- bool mask: `False` 表示被 mask，置为 `-inf`
- 浮点 additive mask: 直接加到 `x` 上

最后沿最后一维执行 softmax。

## 3. 接口规范

```python
cann_bench.causal_softmax_mask(
    Tensor scores,
    Tensor? attention_mask=None,
    float scale=1.0,
    bool causal=True,
) -> Tensor y
```

### 输入参数

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `scores` | `[B, H, Sq, Skv]` | float16 / bfloat16 / float32 | attention score |
| `attention_mask` | 可 broadcast 到 `[B, H, Sq, Skv]` | bool / float16 / bfloat16 / float32 | 可选 mask |
| `scale` | 标量 | float | score 缩放系数 |
| `causal` | 标量 | bool | 是否应用 causal mask |

### 输出

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `y` | 与 `scores` 相同 | 与 `scores` 相同 | softmax 输出 |

### 数据类型

| `scores` dtype | `attention_mask` dtype | 输出 dtype |
|---|---|---|
| float16 | bool / float16 / float32 | float16 |
| bfloat16 | bool / bfloat16 / float32 | bfloat16 |
| float32 | bool / float32 | float32 |

### 规则与约束

- `scores` 必须是 4D 张量 `[B, H, Sq, Skv]`。
- `attention_mask=None` 时只应用 causal mask 或直接 softmax。
- bool mask 中 `True` 表示保留，`False` 表示 mask。
- additive mask 直接加到 score 上，建议 mask 值在 fp16/bf16 可表示范围内。
- 首版公开 case 控制 `B*H*Sq*Skv`，避免 attention score 中间张量超过 NPU 可执行范围。

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


def _right_aligned_causal_mask(sq: int, skv: int, device) -> torch.Tensor:
    q_pos = torch.arange(sq, device=device).unsqueeze(-1)
    k_pos = torch.arange(skv, device=device).unsqueeze(0)
    offset = skv - sq
    return k_pos <= (q_pos + offset)


def causal_softmax_mask(
    scores: torch.Tensor,
    attention_mask: torch.Tensor = None,
    scale: float = 1.0,
    causal: bool = True,
) -> torch.Tensor:
    """Scale + mask + softmax reference for decoder attention scores."""
    if scores.dim() != 4:
        raise ValueError(f"scores must be 4D [B,H,Sq,Skv], got rank {scores.dim()}")

    out_dtype = scores.dtype
    x = scores.float() * float(scale)
    sq, skv = scores.shape[-2], scores.shape[-1]

    if causal:
        keep = _right_aligned_causal_mask(sq, skv, scores.device)
        x = x.masked_fill(~keep.view(1, 1, sq, skv), float("-inf"))

    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            x = x.masked_fill(~attention_mask, float("-inf"))
        else:
            x = x + attention_mask.float()

    y = torch.softmax(x, dim=-1)
    return y.to(out_dtype)
```
