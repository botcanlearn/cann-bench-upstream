# UpsampleNearest2D 算子 API 描述

## 1. 算子简介

`UpsampleNearest2D` 对 NCHW 特征图执行 2D nearest neighbor 上采样，用于补充 research benchmark 中的模型前后处理与表格特征路径覆盖。

**算子特征**：

- 难度等级：L2（Transform）
- x: [B,C,H,W], y: [B,C,out_h,out_w]
- 输出 shape 固定，便于精度比较

## 2. 算子定义

```text
y[b,c,oh,ow] = x[b,c,floor(oh/scale_h),floor(ow/scale_w)]
```

## 3. 接口规范

```python
cann_bench.upsample_nearest2d(Tensor x, int scale_h=2, int scale_w=2) -> Tensor y
```

### 输入参数

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `x` | `[B,C,H,W]` | float16 / bfloat16 / float32 | 输入特征图 |
| `scale_h` | 标量 | int | 高度整数放大倍数 |
| `scale_w` | 标量 | int | 宽度整数放大倍数 |

### 输出

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `y` | `[B,C,H*scale_h,W*scale_w]` | 与主浮点输入相同 | nearest 上采样输出 |

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


def upsample_nearest2d(x: torch.Tensor, scale_h: int = 2, scale_w: int = 2) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError("x must be [B,C,H,W]")
    if scale_h <= 0 or scale_w <= 0:
        raise ValueError("scale_h/scale_w must be positive")
    y = x.repeat_interleave(scale_h, dim=2).repeat_interleave(scale_w, dim=3)
    return y.contiguous()
```
