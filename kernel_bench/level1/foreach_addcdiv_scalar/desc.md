# ForeachAddcdivScalar 算子 API 描述

## 1. 算子简介

ForeachAddcdivScalar 算子对多个张量列表进行逐元素的加、除、乘复合操作，是优化器（如 Adam）中常用的基础运算。

**主要应用场景**：
- Adam / AdamW 优化器的参数更新步骤
- 需要对多组参数同时执行 addcdiv 运算的场景
- 分布式训练中的批量参数更新

**算子特征**：
- 难度等级：L1（FusedComposite）
- 三组 TensorList 输入，逐元素复合运算，输出 TensorList 与输入 shape 一致

## 2. 算子定义

### 数学公式

对列表中第 $i$ 个张量：

$$
y_i = x1_i + \frac{x2_i}{x3_i} \cdot scalar
$$

## 3. 接口规范

### 算子原型

```python
ascend_bench.foreach_addcdiv_scalar(Tensor[] x1, Tensor[] x2, Tensor[] x3, float scalar) -> Tensor[] y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x1 | Tensor[] | 必选 | 第 1 个输入张量列表（TensorList），被加数 |
| x2 | Tensor[] | 必选 | 第 2 个输入张量列表（TensorList），被除数的分子 |
| x3 | Tensor[] | 必选 | 第 3 个输入张量列表（TensorList），被除数的分母 |
| scalar | float | 必选 | 缩放因子 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 与输入 TensorList 各元素 shape 相同 | 与输入 dtype 相同 | 逐元素复合运算结果列表 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

### 规则与约束

- x1、x2、x3 三个 TensorList 长度必须相同
- 对应位置的张量 shape 必须一致
- 列表中各张量的 dtype 须一致
- x3 中的元素不应为零（除以零会产生 inf/nan）

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
from typing import List

def foreach_addcdiv_scalar(
    x1: List[torch.Tensor], x2: List[torch.Tensor], x3: List[torch.Tensor], scalar: float
) -> List[torch.Tensor]:
    """
    对多个张量进行逐元素加、乘、除操作

    公式：y_i = x1_i + (x2_i / x3_i) * scalar

    Args:
        x1: 第 1 个输入张量列表 (TensorList)
        x2: 第 2 个输入张量列表 (TensorList)
        x3: 第 3 个输入张量列表 (TensorList)
        scalar: 缩放因子

    Returns:
        输出张量列表
    """

    y = [x1_i + (x2_i / x3_i) * scalar for x1_i, x2_i, x3_i in zip(x1, x2, x3)]
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import ascend_bench

x1 = [torch.randn(1024, 1024, dtype=torch.float32, device="npu")]
x2 = [torch.randn(1024, 1024, dtype=torch.float32, device="npu")]
x3 = [torch.rand(1024, 1024, dtype=torch.float32, device="npu") + 0.1]  # 避免除零
y = ascend_bench.foreach_addcdiv_scalar(x1, x2, x3, scalar=1.0)
```

### 性能基线参考

基于 cases.yaml 中 20 个测试用例，NPU 上的基准 kernel 执行时间在 12~1351 微秒量级。单张量场景约 13~32 μs，TensorList 长度增加时执行时间近似线性增长（如长度=8 约 138 μs）。

### 相关算子

- **ForeachNorm**：同为 Foreach 系列算子，对 TensorList 进行范数计算
- **ApplyAdamW**（L2）：优化器算子，内部运算逻辑包含 addcdiv 类似计算
