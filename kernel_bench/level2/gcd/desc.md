# Gcd 算子 API 描述

## 1. 算子简介

计算两个整数的最大公约数。

**主要应用场景**：
- 整数约分与最简分数计算
- 数组维度对齐与分块策略中的公因子计算
- 密码学中的模运算与欧几里得算法相关场景

**算子特征**：
- 难度等级：L2（Broadcast）
- 双输入单输出，逐元素运算，输入支持广播

## 2. 算子定义

### 数学公式

$$
y = \gcd(x_1, x_2)
$$

其中 $\gcd(a, b)$ 表示 $a$ 与 $b$ 的最大公约数，即同时整除 $a$ 和 $b$ 的最大正整数。当 $a = b = 0$ 时，$\gcd(0, 0) = 0$。

## 3. 接口规范

### 算子原型

```python
ascend_bench.gcd(Tensor x1, Tensor x2) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x1 | Tensor | 必选 | 第1个输入张量 |
| x2 | Tensor | 必选 | 第2个输入张量 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 广播后的 shape | 与输入一致 | 输出张量，最大公约数 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| int16 | int16 |
| int32 | int32 |
| int64 | int64 |

### 规则与约束

- 两个输入张量的 shape 需满足广播规则，输出 shape 为广播后的 shape
- 两个输入张量的 dtype 必须一致
- 仅支持整数类型（int16、int32、int64）
- Golden 实现中会先将输入转换为 int64 再计算，最终返回 int64 类型结果

## 4. 精度要求

计算结果与 PyTorch Golden 实现逐元素对比，需满足以下误差阈值：

| 数据类型 | 验证方式 | rtol | atol |
|---------|---------|------|------|
| int/uint/bool | 完全相等 | — | — |

**对比公式**：

$$
|output - golden| \leq atol + rtol \times |golden|
$$

## 5. 标准 Golden 代码

```python
import torch

"""
Gcd算子Torch Golden参考实现

计算两个整数的最大公约数
公式: y = gcd(x1, x2)
"""
def gcd(
    x1: torch.Tensor, x2: torch.Tensor
) -> torch.Tensor:
    """
    计算两个整数的最大公约数

    公式: y = gcd(x1, x2)

    Args:
        x1: 第1个输入张量
        x2: 第2个输入张量

    Returns:
        输出张量，最大公约数
    """

    # 转换为int64
    x1_int = x1.to(torch.int64)
    x2_int = x2.to(torch.int64)

    # torch.gcd不支持自动broadcast，需要手动处理
    # 先进行broadcast，再计算gcd
    x1_broadcast, x2_broadcast = torch.broadcast_tensors(x1_int, x2_int)

    y = torch.gcd(x1_broadcast, x2_broadcast)
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import ascend_bench

x1 = torch.randint(-1000, 1000, (1024, 1024), dtype=torch.int32, device="npu")
x2 = torch.randint(-1000, 1000, (1024, 1024), dtype=torch.int32, device="npu")
y = ascend_bench.gcd(x1, x2)

# 广播场景
x1 = torch.randint(-100, 100, (2048, 512), dtype=torch.int16, device="npu")
x2 = torch.randint(-10, 10, (1, 512), dtype=torch.int16, device="npu")
y = ascend_bench.gcd(x1, x2)
```

### 性能基线参考

基于 cases.yaml 中 20 个测试用例，部分用例有基线性能数据（最高约 4869.8 微秒），其余用例的 baseline_perf_us 为 0.0，性能基线数据待完善。

### 相关算子

- **Maximum**：逐元素取最大值，同为双输入广播算子
- **Scatter**：按索引更新张量，涉及整数索引操作
- **ArgMax**：求最大值索引，同为整数输出类算子
