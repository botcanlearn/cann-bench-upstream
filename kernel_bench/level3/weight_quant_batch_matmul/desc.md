# WeightQuantBatchMatmul 算子 API 描述

## 1. 算子简介

权重量化批量矩阵乘法算子。

**主要应用场景**：
- 大语言模型推理中的权重量化加速
- 低精度（INT8/INT4）量化模型的矩阵乘法计算
- 模型压缩与部署场景中的量化矩阵运算

**算子特征**：
- 难度等级：L4（Contraction）
- 三输入（权重 weight、输入 x、偏置 bias）单输出，支持权重反量化后矩阵乘法再量化的完整流程

## 2. 算子定义

### 数学公式

$$
y = \text{quant}(\text{dequant}(weight) \times x + bias)
$$

具体步骤：
1. 对权重矩阵进行反量化（dequant）操作
2. 执行矩阵乘法 $\text{dequant}(weight) \times x$
3. 加上偏置 bias
4. 对结果进行量化（quant）操作

## 3. 接口规范

### 算子原型

```python
ascend_bench.weight_quant_batch_matmul(Tensor weight, Tensor x, Tensor bias, bool transpose_x, bool transpose_weight) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| weight | Tensor | 必选 | 权重矩阵，shape 为 [N, K] |
| x | Tensor | 必选 | 输入矩阵，shape 为 [M, K] |
| bias | Tensor | 必选 | 偏置张量 |
| transpose_x | bool | false | 是否转置 x |
| transpose_weight | bool | false | 是否转置权重 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 由 weight 和 x 的 shape 及 transpose 参数决定 | 与输入 weight 相同 | 输出张量 |

### 数据类型

| 输入 dtype（weight） | 输入 dtype（x） | 输入 dtype（bias） | 输出 dtype |
|--------------------|---------------|-----------------|-----------|
| float16 | float16 | float16 | float16 |
| bfloat16 | bfloat16 | bfloat16 | bfloat16 |
| int8 | int8 | int8 | int8 |
| int4 | int4 | int4 | int4 |

### 规则与约束

- weight 的 shape 为 [N, K]，x 的 shape 为 [M, K]
- 当 transpose_weight=true 时，对 weight 执行最后两维转置后再参与矩阵乘法
- 当 transpose_x=true 时，对 x 执行最后两维转置后再参与矩阵乘法
- 对于 int8/int4 类型的权重，反量化时使用缩放因子进行浮点转换
- 量化阶段使用动态缩放因子并 clamp 到 [-128, 127] 范围

## 4. 精度要求

计算结果与 PyTorch Golden 实现逐元素对比，需满足以下误差阈值：

| 数据类型 | 验证方式 | rtol | atol |
|---------|---------|------|------|
| float16 | 相对误差 | 1e-3 | 1e-3 |
| bfloat16 | 相对误差 | 4e-3 | 4e-3 |
| int8/int4 | 完全相等 | — | — |

**对比公式**：

$$
|output - golden| \leq atol + rtol \times |golden|
$$

## 5. 标准 Golden 代码

```python
import torch

"""
WeightQuantBatchMatmul算子Torch Golden参考实现

权重量化批量矩阵乘法算子
公式: y = quant(dequant(weight) @ x + bias)
"""
def weight_quant_batch_matmul(
    weight: torch.Tensor, x: torch.Tensor, bias: torch.Tensor, transpose_x: bool = False, transpose_weight: bool = False
) -> torch.Tensor:
    """
    权重量化批量矩阵乘法算子
    
    公式: y = quant(dequant(weight) @ x + bias)
    
    Args:
        weight: 权重矩阵
        x: 输入矩阵
        bias: 偏置张量
        transpose_x: 是否转置x
        transpose_weight: 是否转置权重
    
    Returns:
        输出张量
    """

    weight_adj = weight.transpose(-2, -1) if transpose_weight else weight
    x_adj = x.transpose(-2, -1) if transpose_x else x
    
    if weight.dtype in [torch.int8, torch.int4]:
        weight_float = weight.float() * 0.1
    else:
        weight_float = weight.float()
    
    matmul_result = torch.matmul(weight_float, x_adj.float())
    result = matmul_result + bias.float()
    
    scale = 127.0 / result.abs().max()
    y = torch.clamp((result * scale).round(), -128, 127).to(weight.dtype)
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import ascend_bench

weight = torch.randn(256, 512, dtype=torch.float16, device="npu")
x = torch.randn(512, 128, dtype=torch.float16, device="npu")
bias = torch.randn(256, 128, dtype=torch.float16, device="npu")
y = ascend_bench.weight_quant_batch_matmul(weight, x, bias, False, False)
```

### 性能基线参考

基于 cases.yaml 中 20 个测试用例，所有用例的 baseline_perf_us 均为 None，性能基线数据尚未测量。

### 相关算子

- **QuantBatchMatmul**：量化批量矩阵乘法算子，同属量化矩阵运算类别
- **GroupedMatmul**：分组矩阵乘法算子，同属矩阵运算类别
- **DequantSwiGLUQuant**：反量化-SwiGLU-量化融合算子，涉及类似的量化/反量化流程
