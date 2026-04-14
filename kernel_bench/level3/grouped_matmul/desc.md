# GroupedMatmul 算子 API 描述

## 1. 算子简介

分组矩阵乘法算子，对多组输入矩阵分别执行矩阵乘法运算。

**主要应用场景**：
- MoE（Mixture of Experts）模型中多专家的并行矩阵运算
- 多头注意力机制中的分组线性变换
- 批量处理不同大小矩阵的乘法运算

**算子特征**：
- 难度等级：L4（Contraction）
- 双输入单输出，对输入矩阵按组执行矩阵乘法，支持权重转置

## 2. 算子定义

### 数学公式

$$
y[i] = x[i] \times weight[i]
$$

当 `transpose_weight=true` 时：

$$
y[i] = x[i] \times weight[i]^T
$$

## 3. 接口规范

### 算子原型

```python
ascend_bench.grouped_matmul(Tensor x, Tensor weight, int split_item, bool transpose_weight) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入矩阵 |
| weight | Tensor | 必选 | 权重矩阵 |
| split_item | int | 0 | 分组项 |
| transpose_weight | bool | false | 是否转置权重 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 由 x 和 weight 的矩阵乘法决定 | 与输入相同 | 输出张量 |

### 数据类型

| 输入 (x) dtype | 输入 (weight) dtype | 输出 dtype |
|---------------|-------------------|-----------|
| float16 | float16 | float16 |
| bfloat16 | bfloat16 | bfloat16 |
| int8 | int8 | int8 |
| float8 | float8 | float8 |

### 规则与约束

- `x` 的形状为 [M, K]，`weight` 的形状为 [N, K]（transpose_weight=true 时）或 [K, N]（transpose_weight=false 时）
- `x` 和 `weight` 的 dtype 必须一致
- 输出 dtype 与输入 dtype 一致
- `split_item` 用于指定分组方式
- 当 `transpose_weight=true` 时，weight 在乘法前先进行转置

## 4. 精度要求

计算结果与 PyTorch Golden 实现逐元素对比，需满足以下误差阈值：

| 数据类型 | 验证方式 | rtol | atol |
|---------|---------|------|------|
| float16 | 相对误差 | 1e-3 | 1e-3 |
| bfloat16 | 相对误差 | 4e-3 | 4e-3 |
| int8 | 完全相等 | — | — |

**对比公式**：

$$
|output - golden| \leq atol + rtol \times |golden|
$$

## 5. 标准 Golden 代码

```python
import torch

"""
GroupedMatmul算子Torch Golden参考实现

分组矩阵乘法算子
公式: y[i] = x[i] @ weight[i]
"""
def grouped_matmul(
    x: torch.Tensor, weight: torch.Tensor, split_item: int = 0, transpose_weight: bool = False
) -> torch.Tensor:
    """
    分组矩阵乘法算子
    
    公式: y[i] = x[i] @ weight[i]
    
    Args:
        x: 输入矩阵
        weight: 权重矩阵
        split_item: 分组项
        transpose_weight: 是否转置权重
    
    Returns:
        输出张量
    """

    # 分组矩阵乘法
    y = torch.matmul(x, weight.transpose(-2, -1) if transpose_weight else weight)
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import ascend_bench

x = torch.randn(4, 128, 256, dtype=torch.float16, device="npu")
weight = torch.randn(4, 256, 512, dtype=torch.float16, device="npu")
y = ascend_bench.grouped_matmul(x, weight, split_item=0, transpose_weight=False)

# 转置权重
weight_t = torch.randn(4, 512, 256, dtype=torch.float16, device="npu")
y = ascend_bench.grouped_matmul(x, weight_t, split_item=0, transpose_weight=True)
```

### 性能基线参考

基于 cases.yaml 中 20 个测试用例，所有用例的 baseline_perf_us 均为 None，性能基线数据尚未测量。

### 相关算子

- **QuantBatchMatmul**：量化批量矩阵乘法算子，在矩阵乘法基础上增加了量化/反量化操作
- **MoeFinalizeRoutingV2**：MoE 路由合并算子，常与分组矩阵乘法配合使用
- **MoeGatingTopKSoftmax**：MoE 门控网络算子，决定分组矩阵乘法的路由分配
