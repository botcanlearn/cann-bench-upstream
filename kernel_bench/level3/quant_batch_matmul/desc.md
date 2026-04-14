# QuantBatchMatmul 算子 API 描述

## 1. 算子简介

量化批量矩阵乘法算子，执行矩阵乘法后进行量化缩放并加偏置。

**主要应用场景**：
- 大语言模型推理中的低精度量化加速
- INT8/INT4 量化模型中的矩阵运算
- 量化感知训练中的前向计算

**算子特征**：
- 难度等级：L4（Contraction）
- 四输入单输出，支持批量矩阵乘法、量化缩放和偏置加法的融合，支持输入矩阵转置

## 2. 算子定义

### 数学公式

$$
y = \text{dequant}(x_1 \times x_2 \times \text{scale}) + \text{bias}
$$

### 处理流程

1. 若 `transpose_x1=true`，对 $x_1$ 进行转置
2. 若 `transpose_x2=true`，对 $x_2$ 进行转置
3. 执行矩阵乘法：$\text{matmul\_result} = x_1 \times x_2$
4. 应用量化缩放：$\text{scaled\_result} = \text{matmul\_result} \times \text{scale}$
5. 加偏置：$y = \text{scaled\_result} + \text{bias}$

## 3. 接口规范

### 算子原型

```python
ascend_bench.quant_batch_matmul(Tensor x1, Tensor x2, Tensor scale, Tensor bias, int dtype, bool transpose_x1, bool transpose_x2) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x1 | Tensor | 必选 | 第 1 个输入矩阵，形状为 [B, M, K] |
| x2 | Tensor | 必选 | 第 2 个输入矩阵，形状为 [B, K, N] |
| scale | Tensor | 必选 | 量化缩放因子 |
| bias | Tensor | 必选 | 偏置张量 |
| dtype | int | 必选 | 量化数据类型 |
| transpose_x1 | bool | false | 是否转置 x1 |
| transpose_x2 | bool | false | 是否转置 x2 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 由 x1 和 x2 的矩阵乘法决定 | 与 x1 相同 | 输出张量 |

### 数据类型

| 输入 (x1/x2) dtype | 输入 (scale) dtype | 输入 (bias) dtype | 输出 dtype |
|-------------------|------------------|-----------------|-----------|
| int8 | int8 | int8 | int8 |
| int4 | int4 | int4 | int4 |
| float8 | float8 | float8 | float8 |
| float16 | float16 | float16 | float16 |
| bfloat16 | bfloat16 | bfloat16 | bfloat16 |

### 规则与约束

- `x1` 的形状为 [B, M, K]（或 [B, K, M] 当 transpose_x1=true 时），`x2` 的形状为 [B, K, N]（或 [B, N, K] 当 transpose_x2=true 时）
- 批量维度 B 须一致（或满足广播规则）
- `scale` 和 `bias` 的形状需与矩阵乘法输出兼容
- Golden 实现中，中间计算统一提升至 float32 精度，最终转回输入 dtype
- `dtype` 参数用于指定量化数据类型

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
QuantBatchMatmul算子Torch Golden参考实现

量化批量矩阵乘法算子
公式: y = dequant(x1 @ x2 * scale) + bias
"""
def quant_batch_matmul(
    x1: torch.Tensor, x2: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor, dtype: int, transpose_x1: bool = False, transpose_x2: bool = False
) -> torch.Tensor:
    """
    量化批量矩阵乘法算子
    
    公式: y = dequant(x1 @ x2 * scale) + bias
    
    Args:
        x1: 第1个输入矩阵
        x2: 第2个输入矩阵
        scale: 量化缩放因子
        bias: 偏置张量
        dtype: 量化数据类型
        transpose_x1: 是否转置x1
        transpose_x2: 是否转置x2
    
    Returns:
        输出张量
    """

    x1_adj = x1.transpose(-2, -1) if transpose_x1 else x1
    x2_adj = x2.transpose(-2, -1) if transpose_x2 else x2
    
    matmul_result = torch.matmul(x1_adj.float(), x2_adj.float())
    scaled_result = matmul_result * scale.float()
    y = scaled_result + bias.float()
    
    return y.to(x1.dtype)
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import ascend_bench

x1 = torch.randint(-128, 127, (4, 128, 256), dtype=torch.int8, device="npu")
x2 = torch.randint(-128, 127, (4, 256, 512), dtype=torch.int8, device="npu")
scale = torch.ones(1, dtype=torch.int8, device="npu")
bias = torch.zeros(512, dtype=torch.int8, device="npu")
y = ascend_bench.quant_batch_matmul(x1, x2, scale, bias, dtype=0, transpose_x1=False, transpose_x2=False)

# 带转置
x2_t = torch.randint(-128, 127, (4, 512, 256), dtype=torch.int8, device="npu")
y = ascend_bench.quant_batch_matmul(x1, x2_t, scale, bias, dtype=0, transpose_x1=False, transpose_x2=True)
```

### 性能基线参考

基于 cases.yaml 中 20 个测试用例，所有用例的 baseline_perf_us 均为 None，性能基线数据尚未测量。

### 相关算子

- **GroupedMatmul**：分组矩阵乘法算子，不含量化操作
- **MoeFinalizeRoutingV2**：MoE 路由合并算子，涉及矩阵级别的加权求和
- **MoeGatingTopKSoftmax**：MoE 门控网络算子，常与矩阵乘法算子配合使用
