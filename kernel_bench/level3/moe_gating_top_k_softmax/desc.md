# MoeGatingTopKSoftmax 算子 API 描述

## 1. 算子简介

MoE 门控网络中 Softmax 和 TopK 的融合算子，先对输入执行 Softmax 归一化，再取 TopK 个最大值。

**主要应用场景**：
- Mixture of Experts (MoE) 模型中的门控路由选择
- 稀疏激活 Transformer 中的专家选择策略
- MoE 层中 token 到专家的路由分配

**算子特征**：
- 难度等级：L4（FusedComposite）
- 单输入双输出，支持 2D 或 3D 输入，输出 TopK 结果和完整 Softmax 结果

## 2. 算子定义

### 数学公式

$$
y = \text{TopK}(\text{Softmax}(x), k)
$$

### 处理流程

1. 对输入 $x$ 沿最后一维执行 Softmax：$\text{softmax\_out}[i] = \frac{e^{x_i}}{\sum_j e^{x_j}}$
2. 从 Softmax 结果中选取前 $k$ 个最大值作为输出 $y$

## 3. 接口规范

### 算子原型

```python
ascend_bench.moe_gating_top_k_softmax(Tensor x, int k) -> (Tensor y, Tensor softmax_out)
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入张量，支持 2D 或 3D |
| k | int | 必选 | topK 数量 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | (..., k) | 与输入 x 相同 | 输出张量，topK 结果 |
| softmax_out | 与输入 x 相同 | 与输入 x 相同 | Softmax 输出 |

### 数据类型

| 输入 dtype | 输出 (y) dtype | 输出 (softmax_out) dtype |
|-----------|--------------|------------------------|
| float16 | float16 | float16 |
| bfloat16 | bfloat16 | bfloat16 |
| float32 | float32 | float32 |

### 规则与约束

- 输入 `x` 支持 2D 或 3D 张量
- `k` 必须为正整数，且不超过输入最后一维的大小
- Softmax 沿最后一维（dim=-1）计算
- TopK 同样沿最后一维选取
- 输出 `y` 的最后一维大小为 `k`，其余维度与输入一致
- 输出 `softmax_out` 的形状与输入 `x` 完全一致

## 4. 精度要求

计算结果与 PyTorch Golden 实现逐元素对比，需满足以下误差阈值：

| 数据类型 | 验证方式 | rtol | atol |
|---------|---------|------|------|
| float16 | 相对误差 | 1e-3 | 1e-3 |
| float32 | 相对误差 | 1e-4 | 1e-4 |
| bfloat16 | 相对误差 | 4e-3 | 4e-3 |

**对比公式**：

$$
|output - golden| \leq atol + rtol \times |golden|
$$

## 5. 标准 Golden 代码

```python
import torch

"""
MoeGatingTopKSoftmax算子Torch Golden参考实现

MoE门控网络中Softmax和TopK的融合
公式: y = TopK(Softmax(x), k)
"""
def moe_gating_top_k_softmax(
    x: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    MoE门控网络中Softmax和TopK的融合
    
    公式: y = TopK(Softmax(x), k)
    
    Args:
        x: 输入张量
        k: topK数量
    
    Returns:
        y, softmax_out
    """

    softmax_out = torch.nn.functional.softmax(x, dim=-1)
    values, indices = torch.topk(softmax_out, k, dim=-1)
    return values, softmax_out
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import ascend_bench

x = torch.randn(1024, 64, dtype=torch.float16, device="npu")
y, softmax_out = ascend_bench.moe_gating_top_k_softmax(x, k=8)

# 3D 输入
x_3d = torch.randn(4, 1024, 64, dtype=torch.float32, device="npu")
y, softmax_out = ascend_bench.moe_gating_top_k_softmax(x_3d, k=4)
```

### 性能基线参考

基于 cases.yaml 中 20 个测试用例，所有用例的 baseline_perf_us 均为 None，性能基线数据尚未测量。

### 相关算子

- **MoeFinalizeRoutingV2**：MoE 路由合并算子，使用本算子输出的路由权重对专家输出进行加权求和
- **MoeReRouting**：MoE token 重排算子，根据路由结果重新排列 token
- **GroupedMatmul**：分组矩阵乘法算子，在 MoE 流程中紧随门控路由之后执行
