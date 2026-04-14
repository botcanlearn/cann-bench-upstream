# Transpose 算子 API 描述

## 1. 算子简介

对 tensor 的任意维度进行调换。

**主要应用场景**：
- 深度学习中数据格式转换（如 NCHW 与 NHWC 之间的转换）
- 注意力机制中对 Q、K、V 矩阵进行维度交换
- 矩阵运算前的维度调整（如矩阵转置）

**算子特征**：
- 难度等级：L3（LayoutTransform）
- 单输入单输出，支持不超过 8 维的输入，通过 perm 参数指定维度置换顺序

## 2. 算子定义

### 数学公式

$$
y[i_0, ..., i_{n-1}] = x[i_{\text{perm}[0]}, ..., i_{\text{perm}[n-1]}]
$$

其中 perm 为维度置换顺序数组，指定输出张量各维度对应输入张量的哪个维度。

## 3. 接口规范

### 算子原型

```python
ascend_bench.transpose(Tensor x, int[] perm) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入张量，维度不超过 8 维 |
| perm | int[] | 必选 | 维度置换顺序 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 输入 shape 按 perm 重排后的 shape | 与输入 x 相同 | 输出张量，转置后的结果 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |
| int8 | int8 |
| int16 | int16 |
| int32 | int32 |
| int64 | int64 |

### 规则与约束

- 输入维度不超过 8 维
- perm 数组长度必须等于输入维度数，且为 [0, ndim) 的一个排列
- 输出 shape 为输入 shape 按 perm 重排的结果，即 output_shape[i] = input_shape[perm[i]]
- 输出 dtype 与输入 dtype 一致

## 4. 精度要求

计算结果与 PyTorch Golden 实现逐元素对比，需满足以下误差阈值：

| 数据类型 | 验证方式 | rtol | atol |
|---------|---------|------|------|
| float16 | 相对误差 | 1e-3 | 1e-3 |
| float32 | 相对误差 | 1e-4 | 1e-4 |
| bfloat16 | 相对误差 | 4e-3 | 4e-3 |
| int8/int16/int32/int64 | 完全相等 | — | — |

**对比公式**：

$$
|output - golden| \leq atol + rtol \times |golden|
$$

## 5. 标准 Golden 代码

```python
import torch

"""
Transpose算子Torch Golden参考实现

对tensor的任意维度进行调换
公式: y[i0,...,in-1] = x[i_perm[0],...,i_perm[n-1]]
"""
def transpose(
    x: torch.Tensor, perm: list
) -> torch.Tensor:
    """
    对tensor的任意维度进行调换
    
    公式: y[i0,...,in-1] = x[i_perm[0],...,i_perm[n-1]]
    
    Args:
        x: 输入张量
        perm: 维度置换顺序
    
    Returns:
        输出张量，转置后的结果
    """

    y = torch.permute(x, perm)
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import ascend_bench

# 2D 矩阵转置
x = torch.randn(1024, 1024, dtype=torch.float16, device="npu")
y = ascend_bench.transpose(x, [1, 0])

# 4D NCHW 转 NHWC
x = torch.randn(2, 8, 256, 256, dtype=torch.float32, device="npu")
y = ascend_bench.transpose(x, [0, 2, 3, 1])
```

### 性能基线参考

基于 cases.yaml 中 20 个测试用例，所有用例的 baseline_perf_us 均为 None，基线性能尚未测量。测试用例覆盖了 1D 到 5D 的不同维度场景，包含方阵转置、非方阵转置、NCHW 转 NHWC、中间维度交换、全维度反转等多种置换模式，涉及对齐与非对齐 shape、质数维度（如 [363, 367, 373]），以及 float16、float32、bfloat16、int32、int64 等数据类型。

### 相关算子

- **StridedSlice**：使用步长对张量进行多维切片，同属 LayoutTransform 类别
- **Conv2D**：二维卷积算子，在不同数据格式间转换时常需要 Transpose 配合
- **GroupedMatmul**：分组矩阵乘法，计算前可能需要对输入进行维度调整
