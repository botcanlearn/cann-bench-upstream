# GroupedMatmulSwigluQuant 算子 API 描述

## 1. 算子简介

分组矩阵乘法与SwiGLU及量化的融合算子，将矩阵乘法、反量化、SwiGLU 激活和量化四个步骤融合为一个算子执行，减少中间数据搬运开销。

**主要应用场景**：
- 大语言模型中 MoE（Mixture of Experts）结构的 FFN 层融合计算
- 低精度推理场景下的矩阵乘法与激活函数融合
- 需要 int8 量化推理的高性能 Transformer 推理

**算子特征**：
- 难度等级：L4（FusedComposite）
- 双输入（x, weight）单输出，涉及矩阵乘法、反量化、SwiGLU 激活和再量化多步融合
- 输入输出均为 int8 类型，中间计算在浮点域进行

## 2. 算子定义

### 数学公式

$$
y = \text{Quant}(\text{SwiGLU}(\text{Dequant}(\text{Matmul}(x, weight))))
$$

其中各子步骤为：

1. **矩阵乘法**：$M = x \times weight$
2. **反量化**：根据 `dequantMode` 选择反量化方式（模式 0 时乘以缩放因子 0.1，否则直接使用）
3. **SwiGLU 激活**：将反量化结果沿最后一维对半拆分为 $x_{left}$ 和 $x_{right}$，计算 $\text{SiLU}(x_{left}) \times x_{right}$
4. **量化**：将结果量化回 int8 范围 $[-128, 127]$

## 3. 接口规范

### 算子原型

```python
ascend_bench.grouped_matmul_swiglu_quant(Tensor x, Tensor weight, int dequantMode, bool isEnableWeightAssistanceMatrix) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入矩阵，shape 为 [M, K]，dtype 为 int8 |
| weight | Tensor | 必选 | 权重矩阵，shape 为 [E, N/32, K/16, 16, 32]，dtype 为 int8 |
| dequantMode | int | 必选 | 反量化模式 |
| isEnableWeightAssistanceMatrix | bool | false | 是否启用权重辅助矩阵 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 由矩阵乘法及 SwiGLU 拆分决定 | int8 | 融合计算后的量化输出张量 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| int8 | int8 |

### 规则与约束

- 输入 `x` 和 `weight` 均为 int8 类型，中间矩阵乘法在 float32 域进行
- `weight` 的 shape 为 5 维格式 [E, N/32, K/16, 16, 32]，需与 `x` 的 K 维匹配
- `dequantMode` 为 0 时，矩阵乘法结果乘以缩放因子 0.1；其他模式直接使用原始结果
- SwiGLU 要求矩阵乘法输出的最后一维为偶数，以便对半拆分
- 输出量化范围为 $[-128, 127]$，超出范围的值会被截断

## 4. 精度要求

计算结果与 PyTorch Golden 实现逐元素对比，需满足以下误差阈值：

| 数据类型 | 验证方式 | rtol | atol |
|---------|---------|------|------|
| int/uint/bool | 完全相等 | — | — |

**对比公式**：

$$
|output - golden| \leq atol + rtol \times |golden|
$$

## 5. 标准 Golden 代码

```python
import torch

"""
GroupedMatmulSwigluQuant算子Torch Golden参考实现

分组矩阵乘法与SwiGLU及量化的融合
公式: y = SwiGLU(Dequant(Matmul(x, weight)))
"""
def grouped_matmul_swiglu_quant(
    x: torch.Tensor, weight: torch.Tensor, dequantMode: int, isEnableWeightAssistanceMatrix: bool = False
) -> torch.Tensor:
    """
    分组矩阵乘法与SwiGLU及量化的融合
    
    公式: y = SwiGLU(Dequant(Matmul(x, weight)))
    
    Args:
        x: 输入矩阵
        weight: 权重矩阵
        dequantMode: 反量化模式
        isEnableWeightAssistanceMatrix: 是否启用权重辅助矩阵
    
    Returns:
        输出张量
    """

    matmul_result = torch.matmul(x.float(), weight.float())
    
    if dequantMode == 0:
        dequant_result = matmul_result * 0.1
    else:
        dequant_result = matmul_result
    
    half_dim = dequant_result.shape[-1] // 2
    x_left = dequant_result[..., :half_dim]
    x_right = dequant_result[..., half_dim:]
    result = torch.nn.functional.silu(x_left) * x_right
    
    scale = 127.0 / result.abs().max()
    y = torch.clamp((result * scale).round(), -128, 127).to(torch.int8)
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import ascend_bench

x = torch.randint(-128, 127, (64, 256), dtype=torch.int8, device="npu")
weight = torch.randint(-128, 127, (256, 512), dtype=torch.int8, device="npu")
y = ascend_bench.grouped_matmul_swiglu_quant(x, weight, dequantMode=0, isEnableWeightAssistanceMatrix=False)
```

### 性能基线参考

当前暂无测试用例和性能基线数据。

### 相关算子

- **LSTM**：多步融合的循环神经网络算子，同为 L4 级融合算子
- **MlaProlog**：Multi-Head Latent Attention 前处理，同为多步融合算子
- **SparseFlashAttention**：稀疏注意力计算，同为 L4 级融合复合算子
