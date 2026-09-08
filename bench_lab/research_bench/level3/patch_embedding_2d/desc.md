# PatchEmbedding2D 算子 API 描述

## 1. 算子简介

`PatchEmbedding2D` 面向 ViT、OCR 与 VLM 图像输入层，将输入图像按 patch 投影为 token 序列或特征图。该算子覆盖热门图文模型中的图像 patch projection + layout transform 模式。

**算子特征**：

- 难度等级：L3（FusedComposite）
- 输入为 `[B, C, H, W]` 图像和卷积投影权重
- 支持 bias 可选
- 支持输出 `[B, N, E]` token 序列或 `[B, E, Hout, Wout]` 特征图

## 2. 算子定义

```text
proj = conv2d(x, weight, bias, stride=(stride_h, stride_w), padding=(pad_h, pad_w))
if flatten:
    y = proj.flatten(2).transpose(1, 2)
else:
    y = proj
```

其中 `weight.shape = [embed_dim, C, patch_h, patch_w]`。

## 3. 接口规范

```python
cann_bench.patch_embedding_2d(
    Tensor x,
    Tensor weight,
    Tensor? bias=None,
    int stride_h=16,
    int stride_w=16,
    int pad_h=0,
    int pad_w=0,
    bool flatten=True,
) -> Tensor y
```

### 输入参数

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `x` | `[B, C, H, W]` | float16 / bfloat16 / float32 | 输入图像 |
| `weight` | `[E, C, patch_h, patch_w]` | 与 `x` 相同 | patch projection 权重 |
| `bias` | `[E]` 或 None | 与 `x` 相同 | 可选 bias |
| `stride_h/stride_w` | 标量 | int | 卷积 stride |
| `pad_h/pad_w` | 标量 | int | 卷积 padding |
| `flatten` | 标量 | bool | 是否输出 token 序列 |

### 输出

| `flatten` | Shape | dtype |
|---|---|---|
| true | `[B, Hout*Wout, E]` | 与 `x` 相同 |
| false | `[B, E, Hout, Wout]` | 与 `x` 相同 |

### 规则与约束

- `x` 必须为 NCHW 4D 张量。
- `weight` 必须为 4D，且 `weight.shape[1] == x.shape[1]`。
- `stride_h/stride_w > 0`，`pad_h/pad_w >= 0`。
- 首版只覆盖 NCHW，不覆盖 NHWC。
- golden 使用 `torch.nn.functional.conv2d` 表达参考语义。

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


def patch_embedding_2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride_h: int = 16,
    stride_w: int = 16,
    pad_h: int = 0,
    pad_w: int = 0,
    flatten: bool = True,
) -> torch.Tensor:
    if x.dim() != 4 or weight.dim() != 4:
        raise ValueError("x and weight must be 4D tensors")
    if x.shape[1] != weight.shape[1]:
        raise ValueError("weight input channels must match x channels")
    if stride_h <= 0 or stride_w <= 0 or pad_h < 0 or pad_w < 0:
        raise ValueError("stride must be positive and padding must be non-negative")

    out_dtype = x.dtype
    y = torch.nn.functional.conv2d(
        x.float(),
        weight.float(),
        None if bias is None else bias.float(),
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
    )
    if flatten:
        y = y.flatten(2).transpose(1, 2).contiguous()
    return y.to(out_dtype)
```
