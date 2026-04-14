# Conv3DBackpropFilter 算子 API 描述

## 1. 算子简介

Conv3D的filter梯度。

**主要应用场景**：
- 3D 卷积神经网络训练中的反向传播
- 视频理解模型中 Conv3D 层的权重梯度计算
- 医学影像 3D 分割模型的训练过程

**算子特征**：
- 难度等级：L3（Contraction）
- 双输入（输入特征图和输出梯度）单输出（filter 梯度）
- 输入 x 为 [N, C_in, D, H, W] 5维张量

## 2. 算子定义

### 数学公式

$$
y = \text{conv3d\_filter\_grad}(x, \text{grad})
$$

计算 Conv3D 操作中卷积核（filter）的梯度。给定前向传播的输入特征图 $x$ 和来自下游的输出梯度 $\text{grad}$，通过反向传播计算得到 filter 的梯度 $y$。

## 3. 接口规范

### 算子原型

```python
ascend_bench.conv3_d_backprop_filter(Tensor x, Tensor grad, int[] strides, int[] pads, int[] dilations) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入特征图，shape 为 [N, C_in, D, H, W] |
| grad | Tensor | 必选 | 输出梯度 |
| strides | int[] | 必选 | 步长 |
| pads | int[] | 必选 | 填充 |
| dilations | int[] | 必选 | 膨胀率 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 由输入和梯度的 shape 及卷积参数决定 | 与输入 x 相同 | filter梯度 |

### 数据类型

| 输入 (x, grad) dtype | 输出 dtype |
|---------------------|-----------|
| float16 | float16 |
| bfloat16 | bfloat16 |
| float32 | float32 |

### 规则与约束

- x 的 shape 格式为 [N, C_in, D, H, W]
- x 和 grad 的 dtype 须一致
- strides 指定 3D 卷积的步长
- pads 指定填充值
- dilations 指定膨胀率
- 输出 y 为 filter 的梯度，shape 由输入尺寸、梯度尺寸和卷积参数共同决定

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
Conv3DBackpropFilter算子Torch Golden参考实现

Conv3D的filter梯度
公式: y = conv3d_filter_grad(x, grad)
"""
def conv3_d_backprop_filter(
    x: torch.Tensor, grad: torch.Tensor, strides: list, pads: list, dilations: list
) -> torch.Tensor:
    """
    Conv3D的filter梯度
    
    公式: y = conv3d_filter_grad(x, grad)
    
    Args:
        x: 输入特征图
        grad: 输出梯度
        strides: 步长
        pads: 填充
        dilations: 膨胀率
    
    Returns:
        filter梯度
    """

    padding = (pads[0], pads[1], pads[2], pads[3], pads[4])
    stride = (strides[0], strides[1], strides[2])
    dilation = (dilations[0], dilations[1], dilations[2])
    
    y = torch.nn.functional.conv3d(x, grad, bias=None, stride=stride, padding=padding, dilation=dilation)
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import ascend_bench

x = torch.randn(2, 64, 8, 16, 16, dtype=torch.float32, device="npu")
grad = torch.randn(2, 128, 8, 16, 16, dtype=torch.float32, device="npu")

y = ascend_bench.conv3_d_backprop_filter(x, grad, strides=[1, 1, 1], pads=[1, 1, 1, 1, 1], dilations=[1, 1, 1])
```

### 性能基线参考

基于 cases.yaml 中 20 个测试用例，所有用例的 baseline_perf_us 均为 None，性能基线数据尚未测量。

### 相关算子

- **Conv2D**：2D 卷积前向运算
- **DepthwiseConv2D**：深度可分离2D卷积运算
- **AdaptiveAvgPool3D**：3D 自适应平均池化，同样处理5维张量
