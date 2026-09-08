# ImageResizeNormalize 算子 API 描述

## 1. 算子简介

`ImageResizeNormalize` 将 NCHW 图像张量双线性 resize 到固定尺寸后按 channel 做 normalize，用于补充 research benchmark 中的模型前后处理与表格特征路径覆盖。

**算子特征**：

- 难度等级：L2（FusedComposite）
- x: [B,C,H,W], mean/std: [C], y: [B,C,out_h,out_w]
- 输出 shape 固定，便于精度比较

## 2. 算子定义

```text
y = (resize_bilinear(x, [out_h, out_w]) - mean) / std
```

## 3. 接口规范

```python
cann_bench.image_resize_normalize(Tensor x, Tensor mean, Tensor std, int out_h=224, int out_w=224, bool align_corners=False) -> Tensor y
```

### 输入参数

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `x` | `[B,C,H,W]` | float16 / bfloat16 / float32 | 输入图像张量 |
| `mean` | `[C]` | float16 / bfloat16 / float32 | channel mean |
| `std` | `[C]` | float16 / bfloat16 / float32 | channel std |
| `out_h` | 标量 | int | 输出高度 |
| `out_w` | 标量 | int | 输出宽度 |
| `align_corners` | 标量 | bool | 双线性插值是否对齐角点 |

### 输出

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `y` | `[B,C,out_h,out_w]` | 与主浮点输入相同 | resize + normalize 输出 |

### 数据类型

| 输入 dtype | 输出 dtype |
|---|---|
| float16 | float16 |
| bfloat16 | bfloat16 |
| float32 | float32 |

### 规则与约束

- 输入 shape 需满足 `proto.yaml` 中声明的维度关系。
- 所有整数属性需处于有效范围。
- 参考实现仅依赖 PyTorch，输出 dtype 与算子定义保持一致。

## 4. 精度要求

采用[生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)进行验证。

**误差指标**：MERE / MARE；bool 输出按精确匹配处理。

## 5. 标准 Golden 代码

```python
#!/usr/bin/python3
# coding=utf-8

import torch


def image_resize_normalize(
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    out_h: int = 224,
    out_w: int = 224,
    align_corners: bool = False,
) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError("x must be [B,C,H,W]")
    if out_h <= 0 or out_w <= 0:
        raise ValueError("out_h/out_w must be positive")
    channels = x.shape[1]
    if mean.shape != (channels,) or std.shape != (channels,):
        raise ValueError("mean/std must be [C]")
    out_dtype = x.dtype
    resized = torch.nn.functional.interpolate(
        x.float(), size=(out_h, out_w), mode="bilinear", align_corners=align_corners
    )
    denom = torch.clamp(std.float().abs(), min=1.0e-6).reshape(1, channels, 1, 1)
    y = (resized - mean.float().reshape(1, channels, 1, 1)) / denom
    return y.to(out_dtype)
```
