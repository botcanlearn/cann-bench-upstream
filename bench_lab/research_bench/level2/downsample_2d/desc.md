# Downsample2D 算子 API 描述

## 1. 算子简介

`Downsample2D` 对 NCHW 特征图执行 2D average pooling 下采样，用于补充 research benchmark 中的模型前后处理与表格特征路径覆盖。

**算子特征**：

- 难度等级：L2（FusedComposite）
- x: [B,C,H,W], y: pooled NCHW
- 输出 shape 固定，便于精度比较

## 2. 算子定义

```text
y = avg_pool2d(x, kernel_size, stride, padding)
```

## 3. 接口规范

```python
cann_bench.downsample_2d(Tensor x, int kernel_h=2, int kernel_w=2, int stride_h=2, int stride_w=2, int pad_h=0, int pad_w=0) -> Tensor y
```

### 输入参数

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `x` | `[B,C,H,W]` | float16 / bfloat16 / float32 | 输入特征图 |
| `kernel_h` | 标量 | int | 池化窗口高度 |
| `kernel_w` | 标量 | int | 池化窗口宽度 |
| `stride_h` | 标量 | int | 高度方向 stride |
| `stride_w` | 标量 | int | 宽度方向 stride |
| `pad_h` | 标量 | int | 高度方向 padding |
| `pad_w` | 标量 | int | 宽度方向 padding |

### 输出

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `y` | `[B,C,Ho,Wo]` | 与主浮点输入相同 | average pooling 下采样输出 |

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


def downsample_2d(
    x: torch.Tensor,
    kernel_h: int = 2,
    kernel_w: int = 2,
    stride_h: int = 2,
    stride_w: int = 2,
    pad_h: int = 0,
    pad_w: int = 0,
) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError("x must be [B,C,H,W]")
    if min(kernel_h, kernel_w, stride_h, stride_w) <= 0 or min(pad_h, pad_w) < 0:
        raise ValueError("kernel/stride must be positive and padding must be non-negative")
    out_dtype = x.dtype
    y = torch.nn.functional.avg_pool2d(
        x.float(), kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w), padding=(pad_h, pad_w)
    )
    return y.to(out_dtype)
```
