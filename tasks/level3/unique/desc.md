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

采用[生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)进行验证。

**误差指标**：

1. 平均相对误差（MERE）：采样点中相对误差平均值

   $$
   \text{MERE} = \text{avg}(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

2. 最大相对误差（MARE）：采样点中相对误差最大值

   $$
   \text{MARE} = \max(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

**通过标准**：

| 数据类型 | FLOAT16 | BFLOAT16 | FLOAT32 | HiFLOAT32 | FLOAT8 E4M3 | FLOAT8 E5M2 |
|----------|---------|----------|---------|-----------|-------------|-------------|
| **通过阈值(Threshold)** | 2^-10 | 2^-7 | 2^-13 | 2^-11 | 2^-3 | 2^-2 |

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过。


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
