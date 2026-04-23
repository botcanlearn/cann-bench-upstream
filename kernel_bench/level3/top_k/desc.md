# TopK 算子 API 描述

## 1. 算子简介

返回 k 个最大或最小的元素及其索引。

**主要应用场景**：
- 推荐系统中选取得分最高的 k 个候选项
- 分类任务中获取 Top-K 预测类别及其置信度
- 搜索与排序场景中的部分排序加速
- MoE（Mixture of Experts）路由中选取 Top-K 专家

**算子特征**：
- 难度等级：L3（SortSelect）
- 单输入双输出（值和索引），支持 1-8 维输入，支持沿指定维度选取最大或最小的 k 个元素

## 2. 算子定义

### 数学公式

$$
y, idx = \text{topk}(x, k, dim)
$$

沿指定维度 dim 对输入张量 x 进行部分排序，返回前 k 个最大值（当 largest=true）或前 k 个最小值（当 largest=false）及其对应的索引。

## 3. 接口规范

### 算子原型

```python
cann_bench.top_k(Tensor x, int k, int dim, bool largest) -> (Tensor y, Tensor idx)
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入张量，支持 1-8 维 |
| k | int | 必选 | 返回的 topk 数量（取值范围：1 <= k <= dim_size） |
| dim | int | 必选 | 排序维度（取值范围：-ndim ~ ndim-1） |
| largest | bool | true | 是否返回最大值（false 时返回最小值） |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 与输入相同，但 dim 维大小变为 k | 与输入 x 相同 | 输出张量，topk 值 |
| idx | 与 y 相同 | int64 | 输出索引张量（始终为 int64） |

### 数据类型

| 输入 dtype | 输出 dtype（y） | 输出 dtype（idx） |
|-----------|---------------|-----------------|
| int8 | int8 | int64 |
| uint8 | uint8 | int64 |
| int32 | int32 | int64 |
| int64 | int64 | int64 |
| float16 | float16 | int64 |
| float32 | float32 | int64 |
| bfloat16 | bfloat16 | int64 |

### 规则与约束

- 输入支持 1-8 维张量
- k 的取值范围为 1 <= k <= 指定维度的大小
- dim 支持负数索引，取值范围为 -ndim ~ ndim-1
- 当 largest=true 时返回最大的 k 个元素，largest=false 时返回最小的 k 个元素
- 输出 shape 与输入相同，仅 dim 维度大小变为 k

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

"""
TopK算子Torch Golden参考实现

返回k个最大或最小的元素及其索引
公式: y, idx = topk(x, k, dim)
"""
def top_k(
    x: torch.Tensor, k: int, dim: int, largest: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    返回k个最大或最小的元素及其索引
    
    公式: y, idx = topk(x, k, dim)
    
    Args:
        x: 输入张量
        k: 返回的topk数量 (取值范围: 1 <= k <= dim_size)
        dim: 排序维度 (取值范围: -ndim ~ ndim-1)
        largest: 是否返回最大值 (false时返回最小值)
    
    Returns:
        y, idx
    """

    values, indices = torch.topk(x, k=k, dim=dim, largest=largest)
    return values, indices
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.randn(1024, 1024, dtype=torch.float16, device="npu")
y, idx = cann_bench.top_k(x, 10, -1, True)  # 每行取最大的10个元素

x = torch.randn(2, 8, 256, 256, dtype=torch.float32, device="npu")
y, idx = cann_bench.top_k(x, 10, -1, False)  # 每行取最小的10个元素
```

### 性能基线参考

基于 cases.yaml 中 20 个测试用例，所有用例的 baseline_perf_us 均为 None，基线性能尚未测量。测试用例覆盖了 1D 到 5D 的不同维度场景，k 值范围从 1 到 1024（含 k 等于维度大小的极端场景），包含对齐与非对齐 shape、质数维度（如 [3, 7, 11, 13, 1009]）、largest=true/false、sorted=true/false 等多种配置，以及 float16、float32、bfloat16 等数据类型。

### 相关算子

- **Unique**：去除张量中的重复元素，同属 SortSelect 类别
- **MoeGatingTopKSoftmax**：MoE 路由中的 Top-K Softmax 操作，内含 TopK 计算
- **NMS**：非极大值抑制，涉及排序和选择操作
