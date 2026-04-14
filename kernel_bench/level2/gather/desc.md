# Gather 算子 API 描述

## 1. 算子简介

从输入 Tensor 的指定维度按 index 提取元素。

**主要应用场景**：
- 嵌入层（Embedding）的查表操作
- 注意力机制中按索引提取 Key/Value
- 稀疏操作中按索引收集特征

**算子特征**：
- 难度等级：L2（IndexGather）
- 双输入（x 和 index）单输出（y），按索引进行元素提取
- 输入支持 ND 格式，支持任意维度

## 2. 算子定义

### 数学公式

$$
y[i][m][n] = x[index[i]][m][n]
$$

更一般地，对于 `batch_dims=k`，前 k 个维度作为 batch 维度，在第 k 个维度上按 index 进行 gather 操作。

## 3. 接口规范

### 算子原型

```python
ascend_bench.gather(Tensor x, Tensor index, int batch_dims) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入张量 |
| index | Tensor | 必选 | 索引张量 |
| batch_dims | INT64 | 0 | batch 维度数 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 由 index shape 和 x 的非 gather 维度决定 | 与输入 x 相同 | 输出张量，gather 结果 |

### 数据类型

| x dtype | index dtype | 输出 dtype |
|---------|------------|-----------|
| float16 | int32 / int64 | float16 |
| float32 | int32 / int64 | float32 |
| bfloat16 | int32 / int64 | bfloat16 |
| int8 | int32 / int64 | int8 |
| int32 | int32 / int64 | int32 |
| int64 | int32 / int64 | int64 |

### 规则与约束

- 输入支持任意维度的 ND 格式张量
- `batch_dims` 指定 batch 维度数，前 `batch_dims` 个维度作为 batch 维度，x 和 index 在这些维度上的大小必须一致
- index 中的值必须为有效索引，即在 [0, x.shape[batch_dims]) 范围内
- 输出 dtype 与输入 x 的 dtype 一致
- index 张量在 gather 维度之外的维度上，shape 必须与 x 对应维度一致

## 4. 精度要求

计算结果与 PyTorch Golden 实现逐元素对比，需满足以下误差阈值：

| 数据类型 | 验证方式 | rtol | atol |
|---------|---------|------|------|
| float16 | 相对误差 | 1e-3 | 1e-3 |
| float32 | 相对误差 | 1e-4 | 1e-4 |
| bfloat16 | 相对误差 | 4e-3 | 4e-3 |
| int8 | 完全相等 | — | — |
| int32 | 完全相等 | — | — |
| int64 | 完全相等 | — | — |

**对比公式**：

$$
|output - golden| \leq atol + rtol \times |golden|
$$

## 5. 标准 Golden 代码

```python
import torch

"""
Gather算子Torch Golden参考实现

从输入Tensor的指定维度按index提取元素
公式: y[i][m][n] = x[index[i]][m][n]
"""
def gather(
    x: torch.Tensor, index: torch.Tensor, batch_dims: int = 0
) -> torch.Tensor:
    """
    从输入Tensor的指定维度按index提取元素
    
    公式: y[i][m][n] = x[index[i]][m][n]
    
    Args:
        x: 输入张量
        index: 索引张量
        batch_dims: batch维度数
    
    Returns:
        输出张量，gather结果
    """

    y = torch.gather(x, batch_dims, index.long())
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import ascend_bench

x = torch.randn(1024, 1024, dtype=torch.float32, device="npu")
index = torch.randint(0, 1024, (512, 1024), dtype=torch.int32, device="npu")
y = ascend_bench.gather(x, index, batch_dims=0)   # 沿第 0 维 gather

x = torch.randn(128, 128, 64, dtype=torch.float16, device="npu")
index = torch.randint(0, 128, (128, 64, 64), dtype=torch.int64, device="npu")
y = ascend_bench.gather(x, index, batch_dims=1)   # batch_dims=1
```

### 性能基线参考

基于 cases.yaml 中 20 个测试用例，当前所有用例的 baseline_perf_us 均为 0.0，性能基线数据待补充。

### 相关算子

- **Scatter**：Gather 的逆操作，按索引将值写入目标张量
- **ArgMax**：取最大值索引，输出常作为 Gather 的索引输入
- **CrossEntropyLoss**：内部涉及按 target 索引提取 logits 的 gather 操作
