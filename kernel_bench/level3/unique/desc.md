# Unique 算子 API 描述

## 1. 算子简介

去除张量中的重复元素。

**主要应用场景**：
- 数据去重与统计唯一值数量
- 构建词表或标签映射时提取不重复元素
- 稀疏表示和索引压缩中提取唯一键值及逆索引

**算子特征**：
- 难度等级：L3（SortSelect）
- 单输入双输出（唯一值张量和可选的逆索引），支持 ND 格式输入

## 2. 算子定义

### 数学公式

$$
y, inverse = \text{unique}(x, \text{return\_inverse})
$$

对输入张量 x 进行去重操作，返回唯一值张量 y。当 return_inverse=True 时，同时返回逆索引 inverse，满足 $x = y[inverse]$。

## 3. 接口规范

### 算子原型

```python
cann_bench.unique(Tensor x, bool return_inverse) -> (Tensor y, Tensor inverse)
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入张量，支持 ND 格式 |
| return_inverse | bool | false | 是否返回逆索引，用于重建原始张量 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 由唯一值数量决定 | 与输入 x 相同 | 输出张量，唯一值 |
| inverse | 与输入 x 展平后相同 | int64 | 逆索引，满足 x = y[inverse]（当 return_inverse=True 时） |

### 数据类型

| 输入 dtype | 输出 dtype（y） | 输出 dtype（inverse） |
|-----------|---------------|---------------------|
| bfloat16 | bfloat16 | int64 |
| float16 | float16 | int64 |
| float32 | float32 | int64 |
| int8 | int8 | int64 |
| int32 | int32 | int64 |
| int64 | int64 | int64 |
| uint8 | uint8 | int64 |

### 规则与约束

- 输入支持 ND 格式张量，去重前会将输入展平为一维
- 输出唯一值张量 y 的长度取决于输入中不重复元素的数量
- 当 return_inverse=false 时，inverse 输出为 None
- 输出 y 的 dtype 与输入 x 相同，inverse 的 dtype 固定为 int64

## 4. 精度要求

计算结果与 PyTorch Golden 实现逐元素对比，需满足以下误差阈值：

| 数据类型 | 验证方式 | rtol | atol |
|---------|---------|------|------|
| float16 | 相对误差 | 1e-3 | 1e-3 |
| float32 | 相对误差 | 1e-4 | 1e-4 |
| bfloat16 | 相对误差 | 4e-3 | 4e-3 |
| int8/uint8/int32/int64 | 完全相等 | — | — |

**对比公式**：

$$
|output - golden| \leq atol + rtol \times |golden|
$$

## 5. 标准 Golden 代码

```python
import torch

def unique(
    x: torch.Tensor,
    return_inverse: bool = False
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    去除张量中的重复元素

    公式：y, inverse = unique(x, return_inverse)

    Args:
        x: 输入张量
        return_inverse: 是否返回逆索引，用于重建原始张量

    Returns:
        y: 唯一值张量
        inverse: 逆索引，满足 x = y[inverse] (当 return_inverse=True 时)
    """

    if return_inverse:
        y, inverse = torch.unique(x, return_inverse=True)
        return y, inverse
    else:
        y = torch.unique(x, return_inverse=False)
        return y, None
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.tensor([1, 2, 3, 2, 1, 4, 3], dtype=torch.int32, device="npu")
y, inverse = cann_bench.unique(x, True)  # y=[1,2,3,4], inverse=[0,1,2,1,0,3,2]

x = torch.randn(1024, 1024, dtype=torch.float16, device="npu")
y, _ = cann_bench.unique(x, False)  # 仅返回唯一值
```

### 性能基线参考

基于 cases.yaml 中 20 个测试用例，所有用例的 baseline_perf_us 均为 None，基线性能尚未测量。测试用例覆盖了 1D 到 4D 的不同维度场景，包含对齐与非对齐 shape、质数维度（如 [363, 367, 373]），return_inverse 的 true/false 两种模式，以及 float16、float32、bfloat16、int32、int64 等数据类型，并包含零值和特殊值范围输入。

### 相关算子

- **TopK**：返回 k 个最大或最小元素及其索引，同属 SortSelect 类别
- **MoeGatingTopKSoftmax**：MoE 路由中的 Top-K Softmax 操作，涉及排序选择
- **StridedSlice**：多维切片操作，同属数据提取类操作
