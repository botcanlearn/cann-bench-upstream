# Cummin 算子 API 描述

## 1. 算子简介

计算输入张量中的累积最小值。

**主要应用场景**：
- 时间序列分析中的滑动最小值追踪
- 单调约束优化问题中的前缀最小值计算
- 动态规划中的状态转移辅助操作

**算子特征**：
- 难度等级：L3（Reduction）
- 单输入单输出，沿指定轴进行累积归约操作
- 输入输出 shape 相同

## 2. 算子定义

### 数学公式

$$
y[i] = \min(x[0], x[1], \ldots, x[i]) \quad \text{沿指定轴}
$$

即对于输出的第 $i$ 个位置，其值为输入在指定轴上从位置 0 到位置 $i$ 的所有元素中的最小值。

## 3. 接口规范

### 算子原型

```python
ascend_bench.cummin(Tensor x, int axis) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入张量 |
| axis | int64 | 必选 | 计算累积最小值的轴 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 与输入 x 相同 | 与输入 x 相同 | 输出张量，累积最小值 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |
| int32 | int32 |
| bfloat16 | bfloat16 |

### 规则与约束

- 输出 shape 与输入 shape 完全一致
- `axis` 支持负数索引（如 -1 表示最后一维）
- 累积操作沿指定轴按顺序从前到后进行
- 输出 dtype 与输入 dtype 一致

## 4. 精度要求

计算结果与 PyTorch Golden 实现逐元素对比，需满足以下误差阈值：

| 数据类型 | 验证方式 | rtol | atol |
|---------|---------|------|------|
| float16 | 相对误差 | 1e-3 | 1e-3 |
| float32 | 相对误差 | 1e-4 | 1e-4 |
| bfloat16 | 相对误差 | 4e-3 | 4e-3 |
| int32 | 完全相等 | — | — |

**对比公式**：

$$
|output - golden| \leq atol + rtol \times |golden|
$$

## 5. 标准 Golden 代码

```python
import torch

"""
Cummin算子Torch Golden参考实现

计算输入张量中的累积最小值
公式: y[i] = min(x[0], x[1], ..., x[i]) 沿指定轴
"""
def cummin(
    x: torch.Tensor, dim: int
) -> torch.Tensor:
    """
    计算输入张量中的累积最小值

    公式: y[i] = min(x[0], x[1], ..., x[i]) 沿指定轴

    Args:
        x: 输入张量
        dim: 计算累积最小值的轴

    Returns:
        输出张量，累积最小值
    """

    y = torch.cummin(x, dim=dim)[0]
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import ascend_bench

x = torch.randn(1024, 1024, dtype=torch.float32, device="npu")
y = ascend_bench.cummin(x, axis=-1)   # 沿最后一维计算累积最小值

x = torch.randn(2, 8, 256, 256, dtype=torch.float16, device="npu")
y = ascend_bench.cummin(x, axis=2)    # 沿第 2 维计算累积最小值
```

### 性能基线参考

基于 cases.yaml 中 20 个测试用例，所有用例的 baseline_perf_us 均为 None，性能基线数据尚未测量。

### 相关算子

- **ArgMax**：取最大值索引，同为沿指定维度的归约类操作
- **CrossEntropyLoss**：交叉熵损失，涉及 softmax 归约计算
- **UnsortedSegmentSum**：无序分段求和，同为归约类算子
