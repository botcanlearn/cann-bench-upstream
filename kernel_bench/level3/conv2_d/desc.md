# Conv2D 算子 API 描述

## 1. 算子简介

计算2D卷积。

**主要应用场景**：
- 图像分类、目标检测、语义分割等视觉任务的核心运算
- CNN 网络中特征提取的基础模块
- 信号处理中的2D滤波操作

**算子特征**：
- 难度等级：L3（Contraction）
- 三输入（特征图、卷积核、偏置）单输出，支持分组卷积、膨胀卷积
- 输入 x 为 [N, C_in, H, W]，卷积核 filter 为 [C_out, C_in, K_h, K_w]

## 2. 算子定义

### 数学公式

$$
y = \text{CONV}(x, \text{filter}) + \text{bias}
$$

即对输入特征图 $x$ 使用卷积核 $\text{filter}$ 进行2D卷积运算，并加上偏置 $\text{bias}$。卷积运算支持通过 strides 控制步长、pads 控制填充、dilations 控制膨胀率、groups 控制分组数。

## 3. 接口规范

### 算子原型

```python
cann_bench.conv2_d(Tensor x, Tensor filter, Tensor bias, int[] strides, int[] pads, int[] dilations, int groups) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入特征图，shape 为 [N, C_in, H, W] |
| filter | Tensor | 必选 | 卷积核，shape 为 [C_out, C_in, K_h, K_w] |
| bias | Tensor | 必选 | 偏置 |
| strides | int[] | 必选 | 步长 |
| pads | int[] | 必选 | 填充 |
| dilations | int[] | [1, 1] | 膨胀率 |
| groups | int | 1 | 分组数 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 由输入尺寸、卷积核、步长、填充和膨胀率决定 | 与输入 x 相同 | 输出特征图 |

### 数据类型

| 输入 (x, filter, bias) dtype | 输出 dtype |
|-----------------------------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

### 规则与约束

- x 的 shape 格式为 [N, C_in, H, W]
- filter 的 shape 格式为 [C_out, C_in/groups, K_h, K_w]
- x、filter、bias 的 dtype 须一致
- strides 指定卷积的步长
- pads 指定四方向填充 [pad_top, pad_bottom, pad_left, pad_right]
- dilations 指定膨胀率，默认 [1, 1]
- groups 指定分组数，C_in 和 C_out 都须能被 groups 整除

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
Conv2D算子Torch Golden参考实现

计算2D卷积
公式: y = CONV(x, filter) + bias
"""
def conv2_d(
    x: torch.Tensor, filter: torch.Tensor, bias: torch.Tensor, strides: list, pads: list, dilations: list = [1, 1], groups: int = 1
) -> torch.Tensor:
    """
    计算2D卷积
    
    公式: y = CONV(x, filter) + bias
    
    Args:
        x: 输入特征图
        filter: 卷积核
        bias: 偏置
        strides: 步长
        pads: 填充
        dilations: 膨胀率
        groups: 分组数
    
    Returns:
        输出特征图
    """

    padding = (pads[0], pads[1], pads[2], pads[3])
    stride = (strides[0], strides[1])
    dilation = (dilations[0], dilations[1])
    
    y = torch.nn.functional.conv2d(x, filter, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.randn(1, 64, 56, 56, dtype=torch.float16, device="npu")
weight = torch.randn(128, 64, 3, 3, dtype=torch.float16, device="npu")
bias = torch.randn(128, dtype=torch.float16, device="npu")

y = cann_bench.conv2_d(x, weight, bias, strides=[1, 1], pads=[1, 1, 1, 1], dilations=[1, 1], groups=1)
```

### 性能基线参考

基于 cases.yaml 中 20 个测试用例，所有用例的 baseline_perf_us 均为 None，性能基线数据尚未测量。

### 相关算子

- **DepthwiseConv2D**：深度可分离卷积，groups 等于输入通道数的特殊卷积
- **Conv3DBackpropFilter**：3D 卷积的 filter 梯度计算
- **Dilation2D**：2D 形态学膨胀操作，与卷积结构类似但使用 max 代替 sum
