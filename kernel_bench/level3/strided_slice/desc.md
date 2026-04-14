# StridedSlice 算子 API 描述

## 1. 算子简介

使用步长对输入张量进行多维切片，提取子张量。支持 begin_mask、end_mask 控制边界、shrink_axis_mask 收缩维度、new_axis_mask 插入新维度、ellipsis_mask 省略号等功能。

**主要应用场景**：
- 深度学习模型中对特征图进行区域裁剪和下采样
- 序列模型中按步长提取时间步或特征片段
- 数据预处理中对多维张量进行灵活的切片操作
- 模型推理中通过掩码机制实现复杂的维度操控

**算子特征**：
- 难度等级：L3（LayoutTransform）
- 单输入单输出，支持 0-8 维输入，支持负数步长和多种掩码控制

## 2. 算子定义

### 数学公式

$$
y[i,j,k,...] = x[\text{begin}[0]:\text{end}[0]:\text{strides}[0],\ \text{begin}[1]:\text{end}[1]:\text{strides}[1],\ \text{begin}[2]:\text{end}[2]:\text{strides}[2],\ ...]
$$

各掩码参数的作用：
- **begin_mask**：二进制掩码，位 1 表示该维度从 0 开始，忽略 begin 值
- **end_mask**：二进制掩码，位 1 表示该维度切到末尾，忽略 end 值
- **ellipsis_mask**：二进制掩码，位 1 表示该维度使用省略号标记
- **shrink_axis_mask**：二进制掩码，位 1 表示该维度被收缩掉（维度大小为 1）
- **new_axis_mask**：二进制掩码，位 1 表示该位置插入大小为 1 的新维度

## 3. 接口规范

### 算子原型

```python
ascend_bench.strided_slice(Tensor x, int[] begin, int[] end, int[] strides, int begin_mask, int end_mask, int ellipsis_mask, int shrink_axis_mask, int new_axis_mask) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入张量 |
| begin | int[] | 必选 | 切片起始位置数组，长度等于输入维度数 |
| end | int[] | 必选 | 切片结束位置数组，长度等于输入维度数 |
| strides | int[] | 必选 | 切片步长数组，长度等于输入维度数，支持负数步长 |
| begin_mask | int64_t | — | 二进制掩码，位 1 表示该维度从 0 开始，位 0 使用 begin 值 |
| end_mask | int64_t | — | 二进制掩码，位 1 表示该维度切到末尾，位 0 使用 end 值 |
| ellipsis_mask | int64_t | — | 二进制掩码，位 1 表示该维度使用省略号标记 |
| shrink_axis_mask | int64_t | — | 二进制掩码，位 1 表示该维度被收缩掉（维度大小为 1） |
| new_axis_mask | int64_t | — | 二进制掩码，位 1 表示该位置插入大小为 1 的新维度 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 由 begin、end、strides 及各掩码决定 | 与输入 x 相同 | 输出张量，切片结果 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| int8 | int8 |
| uint8 | uint8 |
| int32 | int32 |
| int64 | int64 |
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

### 规则与约束

- 输入支持 0-8 维张量
- begin、end、strides 数组长度必须等于输入维度数
- strides 中每个元素不能为 0，支持负数步长（表示逆序切片）
- begin 和 end 支持负数索引（表示从末尾倒数）
- 各掩码参数以二进制位的形式对应各维度，低位对应低维度
- 输出 dtype 与输入 dtype 一致

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

def strided_slice(
    x: torch.Tensor, begin: list, end: list, strides: list,
    begin_mask: int = 0, end_mask: int = 0, ellipsis_mask: int = 0,
    shrink_axis_mask: int = 0, new_axis_mask: int = 0
) -> torch.Tensor:
    """
    使用步长对输入张量进行多维切片，提取子张量。支持begin_mask、end_mask控制边界、shrink_axis_mask收缩维度、new_axis_mask插入新维度、ellipsis_mask省略号等功能

    公式: y[i,j,k,...] = x[begin[i]:end[i]:strides[i], begin[j]:end[j]:strides[j], begin[k]:end[k]:strides[k], ...]

    Args:
        x: 输入张量
        begin: 切片起始位置数组，长度等于输入维度数
        end: 切片结束位置数组，长度等于输入维度数
        strides: 切片步长数组，长度等于输入维度数，支持负数步长
        begin_mask: begin_mask为二进制掩码，位1表示该维度从0开始，位0使用begin值
        end_mask: end_mask为二进制掩码，位1表示该维度切到末尾，位0使用end值
        ellipsis_mask: ellipsis_mask为二进制掩码，位1表示该维度使用省略号标记
        shrink_axis_mask: shrink_axis_mask为二进制掩码，位1表示该维度被收缩掉（维度大小为1）
        new_axis_mask: new_axis_mask为二进制掩码，位1表示该位置插入大小为1的新维度

    Returns:
        输出张量，切片结果
    """

    slices = [slice(b, e, s) for b, e, s in zip(begin, end, strides)]
    y = x[slices]
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import ascend_bench

x = torch.randn(1024, 1024, dtype=torch.float16, device="npu")
y = ascend_bench.strided_slice(x, [0, 0], [512, 512], [2, 2], 0, 0, 0, 0, 0)

x = torch.randn(2, 8, 256, 256, dtype=torch.float32, device="npu")
y = ascend_bench.strided_slice(x, [0, 0, 0, 0], [-1, -1, 128, 128], [1, 1, 2, 2], 0, 0, 0, 0, 0)
```

### 性能基线参考

基于 cases.yaml 中 20 个测试用例，所有用例的 baseline_perf_us 均为 None，基线性能尚未测量。测试用例覆盖了 1D 到 4D 的不同维度场景，包含对齐与非对齐 shape、质数维度（如 [363, 367, 373]）、不同步长（1 到 4）、局部切片、float16、float32、bfloat16 等数据类型，以及零值和特殊值范围输入。

### 相关算子

- **Transpose**：对张量维度进行调换，同属 LayoutTransform 类别
- **Unique**：去除张量中的重复元素，同属数据重组类操作
- **TopK**：返回 k 个最大或最小元素，同属选择类操作
