# UnsortedSegmentSum 算子 API 描述

## 1. 算子简介

沿 segment_ids 指定的段对数据进行求和。

**主要应用场景**：
- 图神经网络中的节点特征聚合（按邻居分段求和）
- 点云处理中的体素化聚合
- 稀疏特征的按组求和与池化
- 嵌入表梯度的按 ID 累加

**算子特征**：
- 难度等级：L2（ScatterUpdate）
- 双输入单输出，根据 segment_ids 将 data 中的元素按段分组求和

## 2. 算子定义

### 数学公式

$$
y[i] = \sum_{j: \text{segment\_ids}[j] = i} \text{data}[j]
$$

对于每个段 $i \in [0, \text{num\_segments})$，将所有 segment_ids 等于 $i$ 的 data 元素在第 0 维上求和。若某段没有对应的元素，则输出为零。

## 3. 接口规范

### 算子原型

```python
cann_bench.unsorted_segment_sum(Tensor data, Tensor segment_ids, int num_segments) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| data | Tensor | 必选 | 输入数据张量 |
| segment_ids | Tensor | 必选 | 段 ID 张量，值在 [0, num_segments) 范围内 |
| num_segments | int | 必选 | 段数量 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | (num_segments, *data.shape[1:]) | 与 data 相同 | 输出张量，段求和结果 |

### 数据类型

| data dtype | segment_ids dtype | 输出 dtype |
|-----------|------------------|-----------|
| float16 | int32 / int64 | float16 |
| float32 | int32 / int64 | float32 |
| int32 | int32 / int64 | int32 |
| int64 | int32 / int64 | int64 |

### 规则与约束

- segment_ids 的形状必须与 data 的第 0 维大小一致，或与 data 形状完全一致（多维场景）
- segment_ids 中的值必须在 [0, num_segments) 范围内
- 输出的第 0 维大小为 num_segments，其余维度与 data 的后续维度一致
- 若某个段 ID 在 segment_ids 中未出现，对应输出段为全零
- segment_ids 的 dtype 必须为 int32 或 int64
- num_segments 必须为正整数

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

"""
UnsortedSegmentSum算子Torch Golden参考实现

沿segment_ids指定的段对数据进行求和
公式: y[i] = sum(data[j]) where segment_ids[j] == i
"""
def unsorted_segment_sum(
    data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """
    沿segment_ids指定的段对数据进行求和
    
    公式: y[i] = sum(data[j]) where segment_ids[j] == i
    
    Args:
        data: 输入数据张量
        segment_ids: 段ID张量
        num_segments: 段数量
    
    Returns:
        输出张量，段求和结果
    """

    y = torch.zeros(num_segments, *data.shape[1:], dtype=data.dtype, device=data.device)
    for i in range(num_segments):
        mask = (segment_ids == i)
        y[i] = data[mask].sum(dim=0)
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

data = torch.randn(1048576, dtype=torch.float16, device="npu")
segment_ids = torch.randint(0, 1024, (1048576,), dtype=torch.int32, device="npu")
y = cann_bench.unsorted_segment_sum(data, segment_ids, num_segments=1024)

# 2D 数据按段求和
data = torch.randn(1024, 1024, dtype=torch.float32, device="npu")
segment_ids = torch.randint(0, 256, (1024,), dtype=torch.int32, device="npu")
y = cann_bench.unsorted_segment_sum(data, segment_ids, num_segments=256)

# int32 数据类型
data = torch.randint(-1000, 1000, (2048, 512), dtype=torch.int32, device="npu")
segment_ids = torch.randint(0, 512, (2048,), dtype=torch.int32, device="npu")
y = cann_bench.unsorted_segment_sum(data, segment_ids, num_segments=512)
```
