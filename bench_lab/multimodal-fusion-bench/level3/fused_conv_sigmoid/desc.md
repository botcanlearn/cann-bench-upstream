# FusedConvSigmoid 算子

## 算子简介

FusedConvSigmoid 是一个融合算子，将2D卷积操作与sigmoid激活函数融合为一步计算：
`y = sigmoid(conv2d(x, filter, bias))`

融合后避免了中间tensor的内存读写，提升计算效率。

## 算子定义

### 数学公式

```
y = σ(x * filter + bias)
```

其中 `σ(z) = 1 / (1 + exp(-z))` 为sigmoid函数，`*` 表示2D卷积操作。

## 接口规范

```python
def fused_conv_sigmoid(x, filter, bias, strides, pads, dilations=[1,1]):
    """
    Args:
        x: Tensor[N, C_in, H, W] - 输入特征图
        filter: Tensor[C_out, C_in, K_h, K_w] - 卷积核
        bias: Tensor[C_out] - 偏置
        strides: List[int] - [stride_h, stride_w]
        pads: List[int] - [pad_top, pad_bottom, pad_left, pad_right]
        dilations: List[int] - [dilation_h, dilation_w], 默认 [1,1]

    Returns:
        y: Tensor[N, C_out, H_out, W_out] - 输出特征图
    """
```

### 支持的数据类型

| 输入 | 权重 | 偏置 | 输出 |
|------|------|------|------|
| float16 | float16 | float16 | float16 |
| float32 | float32 | float32 | float32 |
| bfloat16 | bfloat16 | bfloat16 | bfloat16 |

### 支持的参数范围

- N: 1~16
- C_in / C_out: 3~1024
- H / W: 8~256
- K_h / K_w: 1~7
- strides: 1~2
- pads: 0~3
- dilations: 1~2

## 精度要求

| dtype | Threshold | 通过条件 |
|-------|-----------|---------|
| float16 | 2^-10 ≈ 0.00098 | MERE < T 且 MARE < 10T |
| bfloat16 | 2^-7 ≈ 0.0078 | 同上 |
| float32 | 2^-13 ≈ 0.000122 | 同上 |

## 标准Golden代码

```python
import torch
import torch.nn.functional as F

def fused_conv_sigmoid(
    x: torch.Tensor, filter: torch.Tensor, bias: torch.Tensor,
    strides: list, pads: list, dilations: list = [1, 1]
) -> torch.Tensor:
    stride = tuple(strides)
    dilation = tuple(dilations)
    if pads[0] == pads[1] and pads[2] == pads[3]:
        padding = (pads[0], pads[2])
    else:
        x = F.pad(x, (pads[2], pads[3], pads[0], pads[1]))
        padding = 0
    conv_out = F.conv2d(x, filter, bias, stride=stride, padding=padding, dilation=dilation)
    return torch.sigmoid(conv_out)
```

## 使用示例

```python
import cann_bench

x = torch.randn(1, 32, 16, 16, dtype=torch.float16, device='npu')
weight = torch.randn(32, 32, 3, 3, dtype=torch.float16, device='npu')
bias = torch.randn(32, dtype=torch.float16, device='npu')
y = cann_bench.fused_conv_sigmoid(x, weight, bias, strides=[1,1], pads=[1,1,1,1], dilations=[1,1])
```
