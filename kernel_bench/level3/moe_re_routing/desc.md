# MoeReRouting 算子 API 描述

## 1. 算子简介

将 token 按照专家顺序重新排列，用于 MoE 模型中的数据重分布。

**主要应用场景**：
- Mixture of Experts (MoE) 模型中 token 到专家的数据分发
- MoE 前馈网络前的 token 重排，使同一专家处理的 token 连续排列
- 分布式 MoE 训练中跨 rank 的 token 重新路由

**算子特征**：
- 难度等级：L3（LayoutTransform）
- 双输入单输出，根据每个 rank 的专家 token 数量对输入 token 进行重新排列

## 2. 算子定义

### 数学公式

$$
y = \text{moe\_rerouting}(x, \text{expert\_token\_num\_per\_rank})
$$

根据 `expert_token_num_per_rank` 指定的每个专家的 token 数量，将输入 token 按专家顺序重新排列。

## 3. 接口规范

### 算子原型

```python
ascend_bench.moe_re_routing(Tensor x, Tensor expert_token_num_per_rank, int expert_token_num_type) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入 token 张量，形状为 [A, H] |
| expert_token_num_per_rank | Tensor | 必选 | 每个 rank 的 token 数量 |
| expert_token_num_type | int | 1 | 专家 token 数量类型 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 与输入 x 相同 | 与输入 x 相同 | 输出张量，重新排列的 token |

### 数据类型

| 输入 (x) dtype | 输入 (expert_token_num_per_rank) dtype | 输出 dtype |
|---------------|--------------------------------------|-----------|
| float16 | float16 | float16 |
| bfloat16 | bfloat16 | bfloat16 |
| int8 | int8 | int8 |

### 规则与约束

- `x` 的形状为 2D [A, H]，其中 A 为 token 数量，H 为隐藏维度
- `expert_token_num_per_rank` 指定每个专家分配的 token 数量
- 所有专家的 token 数量之和应等于输入 token 总数（若不等则进行截断或填充）
- 输出 shape 与输入 shape 一致
- 输出 dtype 与输入 dtype 一致

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
MoeReRouting算子Torch Golden参考实现

将token按照专家顺序重新排列
公式: y = moe_rerouting(x, expert_token_num_per_rank)
"""
def moe_re_routing(
    x: torch.Tensor, expert_token_num_per_rank: torch.Tensor, expert_token_num_type: int = 1
) -> torch.Tensor:
    """
    将token按照专家顺序重新排列
    
    公式: y = moe_rerouting(x, expert_token_num_per_rank)
    
    Args:
        x: 输入token张量
        expert_token_num_per_rank: 每个rank的token数量
        expert_token_num_type: 专家token数量类型
    
    Returns:
        输出张量，重新排列的token
    """

    expert_token_num = expert_token_num_per_rank.tolist()
    if isinstance(expert_token_num[0], list):
        expert_token_num = [item for sublist in expert_token_num for item in sublist]
    
    indices = []
    for num_tokens in expert_token_num:
        num = int(num_tokens)
        indices.extend(range(len(indices), len(indices) + num))
    
    if len(indices) > x.shape[0]:
        indices = indices[:x.shape[0]]
    elif len(indices) < x.shape[0]:
        indices.extend(range(len(indices), x.shape[0]))
    
    y = x[indices]
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import ascend_bench

x = torch.randn(1024, 8, dtype=torch.float16, device="npu")
expert_token_num = torch.tensor([128, 128, 128, 128, 128, 128, 128, 128], dtype=torch.int32, device="npu")
y = ascend_bench.moe_re_routing(x, expert_token_num, expert_token_num_type=1)
```

### 性能基线参考

基于 cases.yaml 中 20 个测试用例，所有用例的 baseline_perf_us 均未测量（None）。测试用例覆盖了 topk=1/2/4/8/16 不同配置、num_experts=4/8/16/32 不同专家数量、float16/float32 数据类型、对齐与非对齐（质数）batch 大小，以及零值和特殊值（inf/-inf）输入等场景。

### 相关算子

- **MoeGatingTopKSoftmax**：MoE 门控网络算子，用于决定 token 路由到哪些专家
- **MoeFinalizeRoutingV2**：MoE 路由合并算子，在专家计算完成后合并输出
- **EmbeddingHashLookupOrInsert**：索引查找类算子，同为基于索引的数据重组
