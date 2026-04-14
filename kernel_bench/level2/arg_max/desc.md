# ArgMax 算子 API 描述

## 1. 算子简介

返回张量在指定维度上的最大值的索引。

**主要应用场景**：
- 分类任务中获取预测类别（取 logits 最大值对应的类别索引）
- Top-1 准确率计算
- 贪心解码（Greedy Decoding）中选择概率最大的 token

**算子特征**：
- 难度等级：L3（SortSelect）
- 单输入单输出，沿指定维度进行归约操作
- 输入支持 0-8 维，输出维度比输入少一维（沿指定 dimension 维度归约）

## 2. 算子定义

### 数学公式

$$
y = \arg\max_{axis=dimension}(x)
$$

即返回输入张量 $x$ 在指定维度 `dimension` 上最大值所在的索引位置。

## 3. 接口规范

### 算子原型

```python
ascend_bench.arg_max(Tensor x, int dimension) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入张量 |
| dimension | int64 | 必选 | 计算 argmax 的维度 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 输入 x 去掉 dimension 维后的 shape | int64 | 输出张量，最大值的索引 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | int64 |
| float32 | int64 |
| bfloat16 | int64 |
| int32 | int64 |
| int64 | int64 |

### 规则与约束

- 输入支持 0-8 维张量
- `dimension` 支持负数索引（如 -1 表示最后一维）
- 输出 dtype 固定为 int64
- 输出 shape 为输入 shape 去掉 `dimension` 维度后的结果
- 当指定维度上存在多个相同的最大值时，返回第一个（最小索引）出现的位置

## 4. 精度要求

计算结果与 PyTorch Golden 实现逐元素对比，需满足以下误差阈值：

| 数据类型 | 验证方式 | rtol | atol |
|---------|---------|------|------|
| int64 | 完全相等 | — | — |

输出为索引值（int64 类型），要求与 Golden 结果完全一致。

**对比公式**：

$$
|output - golden| \leq atol + rtol \times |golden|
$$

## 5. 标准 Golden 代码

```python
import torch

"""
ArgMax 算子 Torch Golden 参考实现

返回张量在指定维度上的最大值的索引
公式: y = argmax(x, axis=dim)
"""
def arg_max(
    x: torch.Tensor, dim: int, keepdims: bool = False
) -> torch.Tensor:
    """
    返回张量在指定维度上的最大值的索引

    公式: y = argmax(x, axis=dim)

    Args:
        x: 输入张量
        dim: 计算 argmax 的维度
        keepdims: 是否保持维度

    Returns:
        输出张量，最大值的索引
    """

    y = torch.argmax(x, dim=dim, keepdim=keepdims)
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import ascend_bench

x = torch.randn(1024, 1024, dtype=torch.float32, device="npu")
y = ascend_bench.arg_max(x, dimension=-1)   # 沿最后一维取 argmax，输出 shape [1024]

x = torch.randn(2, 8, 256, 256, dtype=torch.float16, device="npu")
y = ascend_bench.arg_max(x, dimension=2)    # 沿第 2 维取 argmax，输出 shape [2, 8, 256]
```

### 性能基线参考

基于 cases.yaml 中 20 个测试用例，所有用例的 baseline_perf_us 均为 None，性能基线数据尚未测量。

### 相关算子

- **Cummin**：累积最小值算子，同为沿指定维度的归约操作
- **CrossEntropyLoss**：交叉熵损失，分类任务中常与 ArgMax 配合使用
- **Gather**：按索引提取元素，ArgMax 的输出常作为 Gather 的索引输入
