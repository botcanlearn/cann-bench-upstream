# AdaLayerNormZero 算子 API 描述

## 1. 算子简介

`AdaLayerNormZero` 面向 Diffusion Transformer / DiT 类模型，由条件向量生成 shift、scale 和 gate，对输入 token 执行 LayerNorm 后调制。

**算子特征**：

- 难度等级：L3（FusedComposite）
- 覆盖 LayerNorm + conditioning linear + split + elementwise modulation
- 输出 `(y, gate)`

## 2. 算子定义

```text
normed = layer_norm(x, ln_weight, ln_bias, eps)
params = cond @ proj_weight.T + proj_bias
shift, scale, gate = split(params, 3, dim=-1)
y = normed * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
```

## 3. 接口规范

```python
cann_bench.ada_layer_norm_zero(
    Tensor x,
    Tensor cond,
    Tensor ln_weight,
    Tensor ln_bias,
    Tensor proj_weight,
    Tensor? proj_bias=None,
    float eps=1e-6,
) -> (Tensor y, Tensor gate)
```

### 输入参数

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `x` | `[B, T, D]` | float16 / bfloat16 / float32 | token 输入 |
| `cond` | `[B, C]` | 与 `x` 相同 | 条件向量 |
| `ln_weight` | `[D]` | 与 `x` 相同 | LayerNorm gamma |
| `ln_bias` | `[D]` | 与 `x` 相同 | LayerNorm beta |
| `proj_weight` | `[3D, C]` | 与 `x` 相同 | conditioning linear 权重 |
| `proj_bias` | `[3D]` 或 None | 与 `x` 相同 | conditioning linear bias |
| `eps` | 标量 | float | LayerNorm epsilon，必须大于 0 |

### 输出

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `y` | `[B, T, D]` | 与 `x` 相同 | 调制后的 token |
| `gate` | `[B, D]` | 与 `x` 相同 | gate 输出 |

### 规则与约束

- `x` 必须为 3D `[B, T, D]`。
- `cond.shape[0] == x.shape[0]`。
- `proj_weight.shape == [3D, C]`。
- `proj_bias` 为 None 或 shape `[3D]`。
- `eps > 0`。

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


def ada_layer_norm_zero(
    x: torch.Tensor,
    cond: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    proj_weight: torch.Tensor,
    proj_bias: torch.Tensor = None,
    eps: float = 1e-6,
):
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    if x.dim() != 3 or cond.dim() != 2:
        raise ValueError("x must be [B,T,D] and cond must be [B,C]")
    batch, _, hidden = x.shape
    if cond.shape[0] != batch:
        raise ValueError("cond batch must match x batch")
    if proj_weight.shape[0] != 3 * hidden:
        raise ValueError("proj_weight first dim must be 3 * hidden")

    out_dtype = x.dtype
    normed = torch.nn.functional.layer_norm(x.float(), (hidden,), ln_weight.float(), ln_bias.float(), eps)
    params = torch.nn.functional.linear(cond.float(), proj_weight.float(), None if proj_bias is None else proj_bias.float())
    shift, scale, gate = params.chunk(3, dim=-1)
    y = normed * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    return y.to(out_dtype), gate.to(out_dtype)
```
