# Exp 算子 API 描述

## 1. 算子简介

Exp 算子用于计算输入张量的广义指数函数，支持自定义底数（base）、缩放因子（scale）和偏移量（shift）三个参数，涵盖自然指数、任意底数指数等多种变体。

**主要应用场景**：
- Softmax 中的自然指数计算
- 注意力机制中的指数缩放
- 概率分布与对数域间的转换
- 学习率调度与指数衰减

**算子特征**：
- 难度等级：L1（Elementwise）
- 单输入单输出，逐元素运算，输出 shape 与输入完全一致

## 2. 算子定义

### 数学公式

**通用公式**：

$$
y = e^{(x \cdot scale + shift) \cdot \ln(base)}, \quad base > 0
$$

**自然指数**（当 $base \leq 0$ 时，使用自然底数 $e$）：

$$
y = e^{x \cdot scale + shift}
$$

### 特殊情况

| 条件 | 简化公式 |
|------|---------|
| base ≤ 0, scale=1, shift=0 | $y = e^x$ |
| base > 0, scale=1, shift=0 | $y = base^x$ |
| base=1（任意 scale, shift） | $y = 1$（因 $\ln 1 = 0$） |

## 3. 接口规范

### 算子原型

```python
cann_bench.exp(Tensor x, float base, float scale, float shift) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入张量，支持任意维度 |
| base | float | -1.0 | 指数底数；≤ 0 表示使用自然底数 $e$，> 0 表示自定义底数 |
| scale | float | 1.0 | 输入缩放因子 |
| shift | float | 0.0 | 输入偏移量 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 与输入 x 相同 | 与输入 x 相同 | 指数计算结果 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

### 规则与约束

- 输出 shape 与输入 shape 完全一致，输出 dtype 与输入 dtype 一致
- `base` 参数：≤ 0 时一律视为自然底数 $e$；> 0 时使用该值作为底数
- `x` 支持任意维度（1D ~ 5D 及更高维），不限制具体 shape
- 需注意数值溢出：float16 的有效范围约 [-65504, 65504]，float32 下 $e^x$ 在 $|x| > 88$ 左右可能溢出为 inf

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

def exp(
    x: torch.Tensor, base: float = -1.0, scale: float = 1.0, shift: float = 0.0
) -> torch.Tensor:
    """
    计算输入张量的指数函数

    - base <= 0: y = exp(scale * x + shift)
    - base > 0: y = exp((shift + scale * x) * ln(base))

    Args:
        x: 输入张量
        base: 指数底数，base <= 0 表示使用自然底数 e
        scale: 输入缩放因子
        shift: 输入偏移量

    Returns:
        指数计算结果
    """
    temp = scale * x + shift
    if base > 0:
        temp = temp * torch.log(torch.tensor(base, dtype=x.dtype, device=x.device))
    return torch.exp(temp)
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.randn(1024, 1024, dtype=torch.float16, device="npu")
y = cann_bench.exp(x, base=-1.0, scale=1.0, shift=0.0)  # 自然指数 e^x
y = cann_bench.exp(x, base=2.0, scale=1.0, shift=0.0)   # 2^x
y = cann_bench.exp(x, base=-1.0, scale=2.0, shift=1.0)  # e^(2x+1)
```

### 性能基线参考

基于 cases.yaml 中 20 个测试用例，NPU 上的基准 kernel 执行时间在 13~951 微秒量级，典型对齐场景（如 1M 元素 float32）约 16~21 μs。非对齐质数维度（如 [363, 367, 373]）可能显著影响性能。

### 相关算子

- **Sigmoid**：$y = 1/(1+e^{-x})$，内部依赖指数计算
- **Mish**：$y = x \cdot \tanh(\ln(1+e^x))$，包含 Exp 子运算
- **Gelu**：高斯误差线性单元，近似公式中涉及指数运算
