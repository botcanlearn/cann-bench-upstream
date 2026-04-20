# Mish 算子 API 描述

## 1. 算子简介

Mish 是一种自正则化的非单调神经网络激活函数，具有平滑、非单调的特性，在部分场景下性能优于 ReLU 和 Swish。

**主要应用场景**：
- YOLOv4/v5 等目标检测模型的激活层
- 深层卷积网络中替代 ReLU 的激活函数
- 需要平滑梯度的深度学习模型

**算子特征**：
- 难度等级：L1（Elementwise）
- 单输入单输出，逐元素运算，输出 shape 与输入完全一致
- 支持 0~8 维输入

## 2. 算子定义

### 数学公式

$$
y = x \cdot \tanh(\text{softplus}(x)) = x \cdot \tanh(\ln(1 + e^x))
$$

### 特殊情况

| 输入 | 输出 |
|------|------|
| x = 0 | y = 0 |
| x → +∞ | y → x（趋近恒等） |
| x → -∞ | y → 0 |

## 3. 接口规范

### 算子原型

```python
cann_bench.mish(Tensor x) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入张量，支持 0~8 维 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 与输入 x 相同 | 与输入 x 相同 | Mish 激活结果 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

### 规则与约束

- 输出 shape 与输入 shape 完全一致，输出 dtype 与输入 dtype 一致
- 无额外属性参数

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

def mish(
    x: torch.Tensor
) -> torch.Tensor:
    """
    自正则化的非单调神经网络激活函数

    公式: y = x * tanh(softplus(x))

    Args:
        x: 输入张量

    Returns:
        输出张量，Mish激活结果
    """

    softplus = torch.nn.functional.softplus(x)
    y = x * torch.tanh(softplus)
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.randn(1024, 1024, dtype=torch.float32, device="npu")
y = cann_bench.mish(x)
```

### 性能基线参考

基于 cases.yaml 中 20 个测试用例，NPU 上的基准 kernel 执行时间在 5~1980 微秒量级，典型对齐场景（如 1M 元素 float32）约 38~71 μs。

### 相关算子

- **Sigmoid**：$y = 1/(1+e^{-x})$，Mish 的 softplus 内部包含指数运算
- **Gelu**：$y = x \cdot \Phi(x)$，同为平滑非单调激活函数
- **Exp**：$y = e^x$，Mish 计算链中的基础运算
