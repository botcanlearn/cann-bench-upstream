# ForeachNorm 算子 API 描述

## 1. 算子简介

ForeachNorm 算子对输入张量列表（TensorList）的每个张量分别进行范数运算，支持多种范数阶数。

**主要应用场景**：
- 梯度裁剪中的梯度范数计算
- 优化器中的参数范数监控
- 模型正则化中的权重范数约束

**算子特征**：
- 难度等级：L1（Reduction）
- 输入为张量列表，对每个张量独立计算范数，输出为标量张量列表
- 支持 ND 格式输入

## 2. 算子定义

### 数学公式

**通用 p 范数**：

$$
y = \left(\sum_i |x_i|^p\right)^{1/p}
$$

### 常见范数

| 范数阶数 (scalar) | 公式 | 含义 |
|-------------------|------|------|
| 0 | $\sum_i \mathbb{1}(x_i \neq 0)$ | L0 范数（非零元素个数） |
| 1 | $\sum_i \|x_i\|$ | L1 范数（绝对值之和） |
| 2 | $\sqrt{\sum_i x_i^2}$ | L2 范数（欧氏距离） |
| inf | $\max_i \|x_i\|$ | 无穷范数（最大绝对值） |

## 3. 接口规范

### 算子原型

```python
cann_bench.foreach_norm(Tensor[] x, float scalar) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor[] | 必选 | 输入张量列表（TensorList） |
| scalar | float | 必选 | 范数阶数 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 每个元素为标量张量 | 与输入 dtype 相同 | 每个输入张量的范数结果列表 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

### 规则与约束

- 输入为 TensorList，列表中每个张量独立计算范数
- 列表中各张量的 dtype 须一致
- `scalar` 支持正数、负数、0、inf 等值
- 负阶范数要求输入元素不为零

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

def foreach_norm(
    x: List[torch.Tensor], scalar: float
) -> List[torch.Tensor]:
    """
    对输入张量列表的每个张量进行范数运算

    公式：y = (sum |x_i|^p)^(1/p)

    Args:
        x: 输入张量列表 (TensorList)
        scalar: 范数阶数

    Returns:
        输出张量列表，每个张量的范数结果
    """

    y = [torch.norm(tensor, p=scalar) for tensor in x]
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

t1 = torch.randn(1024, 1024, dtype=torch.float32, device="npu")
t2 = torch.randn(2048, 512, dtype=torch.float32, device="npu")
y = cann_bench.foreach_norm([t1, t2], scalar=2.0)  # L2 范数
```

### 性能基线参考

基于 cases.yaml 中 20 个测试用例，NPU 上的基准 kernel 执行时间在 6~138 微秒量级。TensorList 长度增加时执行时间近似线性增长（如长度=8 约 57 μs）。

### 相关算子

- **ForeachAddcdivScalar**：同为 Foreach 系列算子，对 TensorList 进行逐元素复合运算
