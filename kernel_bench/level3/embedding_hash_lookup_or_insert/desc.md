# EmbeddingHashLookupOrInsert 算子 API 描述

## 1. 算子简介

Embedding 表查找算子，根据键索引从嵌入表中获取对应的嵌入向量。

**主要应用场景**：
- 自然语言处理中的词嵌入查找
- 推荐系统中的特征嵌入检索
- 大规模稀疏特征的嵌入表示

**算子特征**：
- 难度等级：L3（IndexGather）
- 双输入单输出，根据索引从嵌入表中提取对应行，输出 shape 为 (*key.shape, embedding_dim)

## 2. 算子定义

### 数学公式

$$
y = \text{embedding\_table}[\text{key}]
$$

对于每个索引 $i$ in key，输出 $y[i] = \text{table}[\text{key}[i]]$。

### 特殊情况

- 当指定 `padding_idx` 时，该索引对应的输出向量为全零
- 当指定 `max_norm` 时，对输出 embedding 进行范数裁剪，使其 L2 范数不超过 `max_norm`

## 3. 接口规范

### 算子原型

```python
ascend_bench.embedding_hash_lookup_or_insert(Tensor table, Tensor key, int64? padding_idx=None, float? max_norm=None) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| table | Tensor | 必选 | 嵌入表权重矩阵，形状为 (num_embeddings, embedding_dim) |
| key | Tensor | 必选 | 查找的键索引，形状任意 |
| padding_idx | int64 | None | 填充索引，如果指定，该索引对应的输出为零 |
| max_norm | float | None | 如果指定，对输出 embedding 进行范数裁剪 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | (*key.shape, embedding_dim) | 与 table 相同 | 输出张量，查找到的嵌入向量 |

### 数据类型

| 输入 (table) dtype | 输入 (key) dtype | 输出 dtype |
|-------------------|-----------------|-----------|
| float32 | int64 | float32 |
| float16 | int64 | float16 |
| bfloat16 | int64 | bfloat16 |

### 规则与约束

- `table` 的形状必须为 2D：(num_embeddings, embedding_dim)
- `key` 中的索引值必须在 [0, num_embeddings) 范围内
- `key` 支持任意形状（1D、2D、3D 等），输出 shape 为 (*key.shape, embedding_dim)
- 输出 dtype 与 `table` 的 dtype 一致
- `padding_idx` 为可选参数，指定时该索引对应的输出向量全为零
- `max_norm` 为可选参数，指定时对输出 embedding 进行 L2 范数裁剪

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
from typing import Optional

"""
EmbeddingHashLookupOrInsert 算子 Torch Golden 参考实现

Embedding 表查找算子，根据键索引从嵌入表中获取对应的嵌入向量
公式：y = embedding_table[key]
"""
def embedding_hash_lookup_or_insert(
    table: torch.Tensor,
    key: torch.Tensor,
    padding_idx: Optional[int] = None,
    max_norm: Optional[float] = None
) -> torch.Tensor:
    """
    Embedding 表查找算子

    根据键索引从嵌入表中获取对应的嵌入向量

    Args:
        table: 嵌入表权重矩阵，形状为 (num_embeddings, embedding_dim)
        key: 查找的键索引，形状任意
        padding_idx: 填充索引，如果指定，该索引对应的输出为零（可选）
        max_norm: 如果指定，对输出 embedding 进行范数裁剪（可选）

    Returns:
        输出张量，查找到的嵌入向量，形状为 (*key.shape, embedding_dim)
    """
    # 使用 PyTorch 的 embedding 查找
    y = torch.nn.functional.embedding(key, table, padding_idx=padding_idx, max_norm=max_norm)
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import ascend_bench

table = torch.randn(10000, 128, dtype=torch.float32, device="npu")
key = torch.randint(0, 10000, (1024,), dtype=torch.int64, device="npu")
y = ascend_bench.embedding_hash_lookup_or_insert(table, key)  # 基础查找
y = ascend_bench.embedding_hash_lookup_or_insert(table, key, padding_idx=0)  # 带填充索引
y = ascend_bench.embedding_hash_lookup_or_insert(table, key, max_norm=1.0)  # 带范数裁剪
```

### 性能基线参考

基于 cases.yaml 中 20 个测试用例，所有用例的 baseline_perf_us 均未测量（None）。测试用例覆盖了 1D/2D/3D key 形状、对齐与非对齐维度、float16/float32/bfloat16 数据类型，以及 padding_idx 和 max_norm 属性的各种组合。

### 相关算子

- **MoeReRouting**：同为 L3 级别的索引重排算子，涉及基于索引的数据重组
- **NMS**：同为 L3 级别的选择类算子，涉及基于索引的数据筛选
- **GroupedMatmul**：涉及分组数据的批量处理
