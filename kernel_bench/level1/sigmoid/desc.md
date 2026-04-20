# Sigmoid 算子 API 描述

## 1. 算子简介

Sigmoid 算子对输入张量完成 Sigmoid 运算，将任意实数映射到 (0, 1) 区间。

**主要应用场景**：
- 二分类任务的输出层激活
- 门控机制（LSTM、GRU 等 RNN 中的门控信号）
- 注意力权重计算
- 概率输出的归一化

**算子特征**：
- 难度等级：L1（Elementwise）
- 单输入单输出，逐元素运算，输出 shape 与输入完全一致
- 支持 ND 格式输入

## 2. 算子定义

### 数学公式

$$
y = \frac{1}{1 + e^{-x}}
$$

### 特殊情况

| 输入 | 输出 |
|------|------|
| x = 0 | y = 0.5 |
| x → +∞ | y → 1 |
| x → -∞ | y → 0 |

## 3. 接口规范

### 算子原型

```python
cann_bench.sigmoid(Tensor x) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入张量，支持 ND 格式 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 与输入 x 相同 | 与输入 x 相同 | Sigmoid 激活结果 |

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

def sigmoid(
    x: torch.Tensor
) -> torch.Tensor:
    """
    对输入Tensor完成Sigmoid运算

    公式: y = 1 / (1 + e^(-x))

    Args:
        x: 输入张量

    Returns:
        输出张量，Sigmoid激活结果
    """

    y = torch.sigmoid(x)
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.randn(1024, 1024, dtype=torch.float32, device="npu")
y = cann_bench.sigmoid(x)
```

### 性能基线参考

基于 cases.yaml 中 20 个测试用例，NPU 上的基准 kernel 执行时间在 4~313 微秒量级，典型对齐场景（如 1M 元素 float32）约 4~7 μs。

### 相关算子

- **Exp**：$y = e^x$，Sigmoid 内部依赖指数计算
- **Mish**：$y = x \cdot \tanh(\text{softplus}(x))$，内部包含 Sigmoid 相关运算
- **SwiGLU**：采用 Swish（$x \cdot \sigma(\beta x)$）作为激活函数，直接调用 Sigmoid
