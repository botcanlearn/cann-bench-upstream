# MaskedScale 算子 API 描述

## 1. 算子简介

MaskedScale 算子对输入张量进行掩码缩放操作，支持输入张量 x 和掩码张量 mask 的不同数据类型组合。

**主要应用场景**：
- Transformer 中的注意力掩码缩放
- Dropout 的掩码乘法实现
- 条件计算中的选择性缩放

**算子特征**：
- 难度等级：L1（MaskPredicate）
- 双输入单输出，逐元素运算，输入 x、mask 和输出 y 的 shape 需一致

## 2. 算子定义

### 数学公式

$$
y = x \cdot mask \cdot scale
$$

## 3. 接口规范

### 算子原型

```python
ascend_bench.masked_scale(Tensor x, Tensor mask, float scale) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入张量 |
| mask | Tensor | 必选 | 掩码张量，shape 须与 x 一致 |
| scale | float | 1.0 | 缩放因子 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 与输入 x 相同 | 与输入 x 相同 | 掩码缩放结果 |

### 数据类型

x 与 mask 支持不同 dtype 的组合：

| x dtype | mask dtype | y dtype |
|---------|-----------|---------|
| float16 | int8 / uint8 / float16 / float32 | float16 |
| bfloat16 | int8 / uint8 / float16 / float32 | bfloat16 |
| float32 | int8 / uint8 / float16 / float32 | float32 |

### 规则与约束

- 输入 x、mask 的 shape 必须完全一致，输出 y 的 shape 也与之相同
- 输出 dtype 与输入 x 的 dtype 一致
- mask 通常取值 0 或 1，但不限于此（也支持连续值掩码）

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

def masked_scale(
    x: torch.Tensor, mask: torch.Tensor, scale: float = 1.0
) -> torch.Tensor:
    """
    对输入张量进行掩码缩放，支持x和mask的不同数据类型组合

    公式: y = x * mask * scale

    Args:
        x: 输入张量
        mask: 掩码张量
        scale: 缩放因子

    Returns:
        掩码缩放结果
    """

    y = x * mask * scale
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import ascend_bench

x = torch.randn(1024, 1024, dtype=torch.float16, device="npu")
mask = torch.randint(0, 2, (1024, 1024), dtype=torch.int8, device="npu")
y = ascend_bench.masked_scale(x, mask, scale=2.0)
```

### 性能基线参考

基于 cases.yaml 中 20 个测试用例，NPU 上的基准 kernel 执行时间在 7~58 微秒量级，典型对齐场景（如 1M 元素 float32）约 10~19 μs。

### 相关算子

- **Exp**：$y = e^x$，同为 L1 级别 Elementwise 算子
- **Sigmoid**：$y = 1/(1+e^{-x})$，可与 MaskedScale 组合用于门控掩码
