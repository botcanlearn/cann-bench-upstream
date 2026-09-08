# WindowAttention2D 算子 API 描述

## 1. 算子简介

`WindowAttention2D` 面向 OCR、Detection 和 ViT/Swin 类视觉模型，将 `[B,H,W,C]` 2D token 按窗口划分，在每个窗口内执行多头 attention，并还原到原始 2D 排布。

**主要应用场景**：
- Swin Transformer window attention
- OCR / detection backbone 局部窗口 token 混合
- 视觉模型中 window partition + attention + reverse window 融合

**算子特征**：
- 难度等级：L4（FusedComposite）
- 输入为 NHWC 2D token
- 支持 H/W 非整除 window 时右侧/下侧补零
- 支持可选 QKV bias 和 position bias

## 2. 算子定义

### 数学公式

```text
windows = partition(x, window_h, window_w)
q, k, v = linear(windows, qkv_weight, qkv_bias).split(3)
y_window = softmax(q @ k.T / sqrt(head_dim) + position_bias) @ v
y = reverse_windows(y_window)
```

## 3. 接口规范

### 算子原型

```python
cann_bench.window_attention_2d(
    Tensor x,
    Tensor qkv_weight,
    Tensor? qkv_bias=None,
    Tensor? position_bias=None,
    int window_h=7,
    int window_w=7,
    int num_heads=4,
) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | Shape | dtype | 描述 |
|------|------|-------|-------|------|
| `x` | Tensor | `[B, H, W, C]` | float16 / bfloat16 / float32 | NHWC 2D token 输入 |
| `qkv_weight` | Tensor | `[3C, C]` | 与 `x` 相同 | QKV projection 权重 |
| `qkv_bias` | Tensor / None | `[3C]` | 与 `x` 相同 | 可选 QKV bias |
| `position_bias` | Tensor / None | `[num_heads, N, N]` | 与 `x` 相同 | 可选窗口内相对位置偏置，`N=window_h*window_w` |
| `window_h/window_w` | int | - | - | 窗口大小 |
| `num_heads` | int | - | - | attention head 数 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| `y` | `[B, H, W, C]` | 与 `x` 相同 | window attention 输出 |

### 规则与约束

- `x` 必须为 4D NHWC。
- `C % num_heads == 0`。
- `qkv_weight.shape == [3C, C]`。
- `qkv_bias` 为 None 或 `[3C]`。
- `position_bias` 为 None 或 `[num_heads, window_h*window_w, window_h*window_w]`。
- `window_h/window_w/num_heads > 0`。

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `B` | 1 ~ 4 | 小批量视觉特征 |
| `H/W` | 7 ~ 64 | 覆盖整除和非整除窗口 |
| `C` | 32 ~ 256 | 必须能被 `num_heads` 整除 |
| `window_h/window_w` | 4 / 7 / 8 | 首版公开 case 覆盖 |
| `num_heads` | 1 / 2 / 4 / 8 | 首版公开 case 覆盖 |

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


def _partition_windows(x: torch.Tensor, window_h: int, window_w: int):
    batch, height, width, channels = x.shape
    pad_h = (window_h - height % window_h) % window_h
    pad_w = (window_w - width % window_w) % window_w
    x_pad = torch.nn.functional.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    hp, wp = height + pad_h, width + pad_w
    windows = x_pad.reshape(batch, hp // window_h, window_h, wp // window_w, window_w, channels)
    windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.reshape(-1, window_h * window_w, channels), hp, wp


def _reverse_windows(windows: torch.Tensor, batch: int, height: int, width: int, hp: int, wp: int,
                     window_h: int, window_w: int) -> torch.Tensor:
    channels = windows.shape[-1]
    x = windows.reshape(batch, hp // window_h, wp // window_w, window_h, window_w, channels)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(batch, hp, wp, channels)
    return x[:, :height, :width, :].contiguous()


def window_attention_2d(
    x: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor = None,
    position_bias: torch.Tensor = None,
    window_h: int = 7,
    window_w: int = 7,
    num_heads: int = 4,
) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError("x must be [B, H, W, C]")
    if window_h <= 0 or window_w <= 0 or num_heads <= 0:
        raise ValueError("window_h/window_w/num_heads must be positive")
    batch, height, width, channels = x.shape
    if channels % num_heads != 0:
        raise ValueError("channels must be divisible by num_heads")
    if qkv_weight.shape != (3 * channels, channels):
        raise ValueError("qkv_weight must be [3C, C]")
    if qkv_bias is not None and qkv_bias.shape != (3 * channels,):
        raise ValueError("qkv_bias must be [3C]")

    out_dtype = x.dtype
    windows, hp, wp = _partition_windows(x, window_h, window_w)
    qkv = torch.nn.functional.linear(windows.float(), qkv_weight.float(), None if qkv_bias is None else qkv_bias.float())
    num_windows, tokens, _ = qkv.shape
    head_dim = channels // num_heads
    qkv = qkv.reshape(num_windows, tokens, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    scores = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)
    if position_bias is not None:
        if position_bias.shape != (num_heads, tokens, tokens):
            raise ValueError("position_bias must be [num_heads, window_h*window_w, window_h*window_w]")
        scores = scores + position_bias.float().unsqueeze(0)
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v).transpose(1, 2).reshape(num_windows, tokens, channels)
    return _reverse_windows(out.to(out_dtype), batch, height, width, hp, wp, window_h, window_w)
```
