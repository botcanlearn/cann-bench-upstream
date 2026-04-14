# Gelu 算子 API 描述

## 1. 算子简介

Gelu（高斯误差线性单元）是一种广泛应用于 Transformer 架构的激活函数，支持精确计算和 tanh 近似两种模式。

**主要应用场景**：
- BERT、GPT 等 Transformer 模型的前馈网络激活层
- Vision Transformer (ViT) 中的 MLP 模块
- 各类预训练语言模型的中间激活

**算子特征**：
- 难度等级：L1（Elementwise）
- 单输入单输出，逐元素运算，输出 shape 与输入完全一致
- 支持 0~8 维输入

## 2. 算子定义

### 数学公式

**精确模式**（approximate="none"）：

$$
y = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]
$$

**tanh 近似模式**（approximate="tanh"）：

$$
y = 0.5 \cdot x \cdot \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \cdot (x + 0.044715 \cdot x^3)\right)\right)
$$

### 特殊情况

| 输入 | 输出 |
|------|------|
| x = 0 | y = 0 |
| x → +∞ | y → x |
| x → -∞ | y → 0 |

## 3. 接口规范

### 算子原型

```python
ascend_bench.gelu(Tensor x, str approximate="none") -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入张量，支持 0~8 维 |
| approximate | str | "none" | GELU 近似计算算法，可选值：'none'（精确计算）或 'tanh'（tanh 近似） |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 与输入 x 相同 | 与输入 x 相同 | GELU 激活结果 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

### 规则与约束

- 输出 shape 与输入 shape 完全一致，输出 dtype 与输入 dtype 一致
- `approximate` 参数仅支持 "none" 和 "tanh" 两种取值
- 输入支持 0~8 维

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

def gelu(
    x: torch.Tensor,
    approximate: str = "none"
) -> torch.Tensor:
    """
    高斯误差线性单元激活函数

    公式：y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    Args:
        x: 输入张量
        approximate: GELU 近似计算算法，可选值：'none'(精确计算) 或 'tanh'(tanh 近似)

    Returns:
        输出张量，GELU 激活结果
    """

    y = torch.nn.functional.gelu(x, approximate=approximate)
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import ascend_bench

x = torch.randn(1024, 1024, dtype=torch.float32, device="npu")
y = ascend_bench.gelu(x)                          # 精确模式
y = ascend_bench.gelu(x, approximate="tanh")       # tanh 近似模式
```

### 性能基线参考

基于 cases.yaml 中 20 个测试用例，NPU 上的基准 kernel 执行时间在 5~314 微秒量级，典型对齐场景（如 1M 元素 float32）约 5~9 μs。

### 相关算子

- **Sigmoid**：$y = 1/(1+e^{-x})$，同为常用激活函数
- **Mish**：$y = x \cdot \tanh(\text{softplus}(x))$，类似的非单调激活函数
- **SwiGLU**：采用 Swish 激活的 GLU 变体，Swish 与 GELU 性质相近
