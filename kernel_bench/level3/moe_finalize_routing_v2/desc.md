# MoeFinalizeRoutingV2 算子 API 描述

## 1. 算子简介

合并 MoE FFN 的输出结果，使用路由权重对专家输出进行加权求和。

**主要应用场景**：
- Mixture of Experts (MoE) 模型中专家输出的最终聚合
- MoE Transformer 层中多专家结果的加权融合
- 稀疏激活模型中 token 级别的专家输出合并

**算子特征**：
- 难度等级：L3（FusedComposite）
- 多输入单输出，支持 2D 和 3D 专家输出格式，可选 token 索引重排和权重归一化

## 2. 算子定义

### 数学公式

$$
\text{output}[i] = \sum_{k=1}^{topk} \text{routing\_weights}[i, k] \times \text{expert\_outputs}[i, k]
$$

### 处理流程

1. 若 `expert_outputs` 为 2D (num_tokens * topk, hidden_dim)，先 reshape 为 3D (num_tokens, topk, hidden_dim)
2. 若 `renormalize=true`，对 routing_weights 沿最后一维重新归一化
3. 使用路由权重对专家输出进行加权求和
4. 若提供 `sorted_token_indices`，按原始顺序恢复输出

## 3. 接口规范

### 算子原型

```python
ascend_bench.moe_finalize_routing_v2(Tensor expert_outputs, Tensor routing_weights, Tensor? sorted_token_indices=None, bool renormalize=false) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| expert_outputs | Tensor | 必选 | 专家输出张量，形状为 (num_tokens * topk, hidden_dim) 或 (num_tokens, topk, hidden_dim) |
| routing_weights | Tensor | 必选 | 路由权重，形状为 (num_tokens, topk)，经过 softmax 归一化 |
| sorted_token_indices | Tensor | None | 可选，排序后的 token 索引，用于恢复原始顺序 |
| renormalize | bool | false | 是否重新归一化路由权重 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | (num_tokens, hidden_dim) | 与 expert_outputs 相同 | 输出张量，加权求和后的 MoE 输出 |

### 数据类型

| 输入 (expert_outputs) dtype | 输入 (routing_weights) dtype | 输入 (sorted_token_indices) dtype | 输出 dtype |
|---------------------------|---------------------------|--------------------------------|-----------|
| float32 | float32 | int32 / int64 | float32 |
| float16 | float16 | int32 / int64 | float16 |
| bfloat16 | bfloat16 | int32 / int64 | bfloat16 |

### 规则与约束

- `expert_outputs` 支持 2D 和 3D 两种输入格式
- 2D 格式时，第一维应为 num_tokens * topk，会自动 reshape 为 3D
- `routing_weights` 的形状必须为 (num_tokens, topk)
- `sorted_token_indices` 为可选参数，不提供时不进行顺序恢复
- `expert_outputs` 和 `routing_weights` 的浮点 dtype 需一致
- `sorted_token_indices` 的 dtype 为 int32 或 int64

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
MoeFinalizeRoutingV2 算子 Torch Golden 参考实现

合并 MoE FFN 的输出结果，使用路由权重对专家输出进行加权求和
公式：output = sum(routing_weights * expert_outputs)
"""
def moe_finalize_routing_v2(
    expert_outputs: torch.Tensor,
    routing_weights: torch.Tensor,
    sorted_token_indices: torch.Tensor = None,
    num_tokens: int = None,
    renormalize: bool = False
) -> torch.Tensor:
    """
    合并 MoE FFN 的输出结果

    对每个 token，使用路由权重对其分配的专家输出进行加权求和

    Args:
        expert_outputs: 专家输出张量，形状为 (num_tokens * topk, hidden_dim)
                       或 (num_tokens, topk, hidden_dim)
        routing_weights: 路由权重，形状为 (num_tokens, topk)，经过 softmax 归一化
        sorted_token_indices: 可选，排序后的 token 索引，用于恢复原始顺序
        num_tokens: 可选，token 数量
        renormalize: 是否重新归一化权重

    Returns:
        输出张量，形状为 (num_tokens, hidden_dim)
    """

    # 处理输入形状
    if expert_outputs.dim() == 2:
        # 形状：(num_tokens * topk, hidden_dim) -> (num_tokens, topk, hidden_dim)
        if num_tokens is None:
            num_tokens = expert_outputs.shape[0] // routing_weights.shape[1]
        topk = routing_weights.shape[1]
        hidden_dim = expert_outputs.shape[1]
        expert_outputs = expert_outputs.view(num_tokens, topk, hidden_dim)
    elif expert_outputs.dim() == 3:
        # 形状：(num_tokens, topk, hidden_dim)
        num_tokens = expert_outputs.shape[0]
        topk = expert_outputs.shape[1]
        hidden_dim = expert_outputs.shape[2]

    # 验证 sorted_token_indices 是否可用
    if sorted_token_indices is not None and sorted_token_indices.numel() == num_tokens:
        # 检查索引值是否在有效范围内
        max_idx = sorted_token_indices.max().item()
        min_idx = sorted_token_indices.min().item()
        if max_idx < num_tokens and min_idx >= 0:
            # 有效的索引，使用 argsort 恢复原始顺序
            output = (expert_outputs * routing_weights.unsqueeze(-1)).sum(dim=1)
            output = output[torch.argsort(sorted_token_indices)]
            return output

    # 重新归一化权重
    if renormalize:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

    # 加权求和：output[i] = sum_k(routing_weights[i, k] * expert_outputs[i, k])
    output = (expert_outputs * routing_weights.unsqueeze(-1)).sum(dim=1)

    return output
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import ascend_bench

# 3D expert_outputs 输入
expert_outputs = torch.randn(1024, 8, 512, dtype=torch.float16, device="npu")
routing_weights = torch.randn(1024, 8, dtype=torch.float16, device="npu").softmax(dim=-1)
y = ascend_bench.moe_finalize_routing_v2(expert_outputs, routing_weights)

# 带 sorted_token_indices
indices = torch.randperm(8192, dtype=torch.int32, device="npu")
y = ascend_bench.moe_finalize_routing_v2(expert_outputs, routing_weights, sorted_token_indices=indices)

# 带 renormalize
y = ascend_bench.moe_finalize_routing_v2(expert_outputs, routing_weights, renormalize=True)
```

### 性能基线参考

基于 cases.yaml 中 20 个测试用例，所有用例的 baseline_perf_us 均未测量（None）。测试用例覆盖了 2D 和 3D expert_outputs 格式、topk=2/4/8 不同配置、float16/float32/bfloat16 数据类型、对齐与非对齐（质数）维度、零值与特殊值（inf/-inf）输入，以及 renormalize 开关等场景。

### 相关算子

- **MoeGatingTopKSoftmax**：MoE 门控网络中 Softmax 和 TopK 的融合算子，用于生成路由权重
- **MoeReRouting**：MoE token 重排算子，将 token 按专家顺序重新排列
- **GroupedMatmul**：分组矩阵乘法算子，常用于 MoE 中各专家的前馈计算
