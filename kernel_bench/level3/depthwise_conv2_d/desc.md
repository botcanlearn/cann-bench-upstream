# DepthwiseConv2D 算子 API 描述

## 1. 算子简介

二维深度卷积运算。

**主要应用场景**：
- MobileNet、EfficientNet 等轻量级网络中的深度可分离卷积
- 边缘设备上的高效特征提取
- 图像处理中的逐通道滤波操作

**算子特征**：
- 难度等级：L3（Contraction）
- 三输入（特征图、卷积核、偏置）单输出，支持分组卷积和膨胀卷积
- 输入 x 为 [N, C_in, H, W]，卷积核 weight 为 [C_out, 1, K_h, K_w]

## 2. 算子定义

### 数学公式

$$
y = \text{bias} + \text{weight} * x
$$

深度卷积对输入的每个通道分别使用独立的卷积核进行卷积运算，再加上偏置。与标准卷积不同，深度卷积中 groups 等于输入通道数，每个通道独立计算。

## 3. 接口规范

### 算子原型

```python
cann_bench.depthwise_conv2_d(Tensor x, Tensor weight, Tensor bias, int[] kernelSize, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入特征图，shape 为 [N, C_in, H, W] |
| weight | Tensor | 必选 | 卷积核，shape 为 [C_out, 1, K_h, K_w] |
| bias | Tensor | 必选 | 偏置 |
| kernelSize | int[] | 必选 | 卷积核大小 |
| stride | int[] | 必选 | 步长 |
| padding | int[] | 必选 | 填充 |
| dilation | int[] | 必选 | 膨胀率 |
| groups | int | 必选 | 分组数 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 由输入尺寸、卷积核、步长、填充和膨胀率决定 | 与输入 x 相同 | 输出特征图 |

### 数据类型

| 输入 (x, weight, bias) dtype | 输出 dtype |
|-----------------------------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

### 规则与约束

- x 的 shape 格式为 [N, C_in, H, W]
- weight 的 shape 格式为 [C_out, 1, K_h, K_w]，每个通道使用独立的卷积核
- x、weight、bias 的 dtype 须一致
- kernelSize 指定卷积核大小
- stride 指定卷积步长
- padding 指定填充
- dilation 指定膨胀率
- groups 指定分组数，深度卷积中通常 groups = C_in

## 4. 精度要求

计算结果与 PyTorch Golden 实现逐元素对比，需满足以下误差阈值：

| 数据类型 | 验证方式 | rtol | atol |
|---------|---------|------|------|
| float16 | 相对误差 | 1e-3 | 1e-3 |
| float32 | 相对误差 | 1e-4 | 1e-4 |
| bfloat16 | 相对误差 | 4e-3 | 4e-3 |

**对比公式**：

$$
|output - golden| \leq atol + rtol \times |golden|
$$

## 5. 标准 Golden 代码

```python
import torch

"""
DepthwiseConv2D算子Torch Golden参考实现

二维深度卷积运算
公式: y = bias + weight * x
"""
def depthwise_conv2_d(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, kernelSize: list, stride: list, padding: list, dilation: list, groups: int
) -> torch.Tensor:
    """
    二维深度卷积运算
    
    公式: y = bias + weight * x
    
    Args:
        x: 输入特征图
        weight: 卷积核
        bias: 偏置
        kernelSize: 卷积核大小
        stride: 步长
        padding: 填充
        dilation: 膨胀率
        groups: 分组数
    
    Returns:
        输出特征图
    """

    stride_val = (stride[0], stride[1])
    padding_val = (padding[0], padding[1])
    dilation_val = (dilation[0], dilation[1])
    
    y = torch.nn.functional.conv2d(x, weight, bias, stride=stride_val, padding=padding_val, dilation=dilation_val, groups=groups)
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.randn(1, 64, 56, 56, dtype=torch.float16, device="npu")
weight = torch.randn(64, 1, 3, 3, dtype=torch.float16, device="npu")
bias = torch.randn(64, dtype=torch.float16, device="npu")

y = cann_bench.depthwise_conv2_d(x, weight, bias, kernelSize=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=64)
```

### 性能基线参考

基于 cases.yaml 中 20 个测试用例，所有用例的 baseline_perf_us 均为 None，性能基线数据尚未测量。

### 相关算子

- **Conv2D**：标准2D卷积运算，深度卷积是其 groups = C_in 的特殊形式
- **Dilation2D**：2D 形态学膨胀操作，同样使用滑动窗口处理2D输入
- **Conv3DBackpropFilter**：3D 卷积的 filter 梯度计算
