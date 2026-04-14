# SparseFlashAttention 算子 API 描述

## 1. 算子简介

大序列长度推理场景的高效注意力计算算子，基于稀疏分块策略降低标准注意力机制的计算和内存开销，支持可配置的稀疏块大小和查询布局格式。

**主要应用场景**：
- 大语言模型长序列推理中的高效注意力计算
- 长文本理解与生成任务中降低注意力计算复杂度
- 需要稀疏注意力模式的 Transformer 推理加速

**算子特征**：
- 难度等级：L4（FusedComposite）
- 三输入（query, key, value）单输出，融合缩放点积注意力与 softmax 计算
- 支持可配置的稀疏块大小和查询布局格式

## 2. 算子定义

### 数学公式

$$
y = \text{softmax}\left(\frac{Q \times K^T}{\sqrt{d}}\right) \times V
$$

其中：
- $Q$ 为查询张量，$K$ 为键张量，$V$ 为值张量
- $\sqrt{d}$ 为缩放因子（由 `scaleValue` 参数指定）
- softmax 沿最后一维计算

具体子步骤：
1. **缩放点积**：$\text{scores} = Q \times K^T \times \text{scaleValue}$
2. **Softmax 归一化**：$\text{attn\_weights} = \text{softmax}(\text{scores}, \text{dim}=-1)$
3. **加权求和**：$y = \text{attn\_weights} \times V$

## 3. 接口规范

### 算子原型

```python
ascend_bench.sparse_flash_attention(Tensor query, Tensor key, Tensor value, float scaleValue, int sparseBlockSize, str layoutQuery) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| query | Tensor | 必选 | 查询张量，shape 为 [B, S1, N1, D]（BSND 布局） |
| key | Tensor | 必选 | 键张量 |
| value | Tensor | 必选 | 值张量 |
| scaleValue | float | 必选 | 缩放因子 |
| sparseBlockSize | int | 必选 | 稀疏块大小 |
| layoutQuery | str | "BSND" | 查询布局格式 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 与 query 相同 | 与输入 query 相同 | 注意力计算输出张量 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| bfloat16 | bfloat16 |
| float16 | float16 |

### 规则与约束

- 所有输入 Tensor（query, key, value）的 dtype 必须一致
- `query` 的默认布局为 BSND：[Batch, SeqLen, NumHeads, HeadDim]
- `key` 和 `value` 的序列长度维度可以与 `query` 不同（支持 cross-attention 场景）
- `scaleValue` 通常设置为 $1/\sqrt{d}$，其中 $d$ 为 head 维度
- `sparseBlockSize` 控制稀疏注意力的分块粒度
- `layoutQuery` 指定张量布局格式，默认为 "BSND"

## 4. 精度要求

计算结果与 PyTorch Golden 实现逐元素对比，需满足以下误差阈值：

| 数据类型 | 验证方式 | rtol | atol |
|---------|---------|------|------|
| float16 | 相对误差 | 1e-3 | 1e-3 |
| bfloat16 | 相对误差 | 4e-3 | 4e-3 |

**对比公式**：

$$
|output - golden| \leq atol + rtol \times |golden|
$$

## 5. 标准 Golden 代码

```python
import torch

"""
SparseFlashAttention算子Torch Golden参考实现

大序列长度推理场景的高效注意力计算
公式: y = softmax(Q @ K^T / sqrt(d)) @ V
"""
def sparse_flash_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scaleValue: float, sparseBlockSize: int, layoutQuery: str = 'BSND'
) -> torch.Tensor:
    """
    大序列长度推理场景的高效注意力计算
    
    公式: y = softmax(Q @ K^T / sqrt(d)) @ V
    
    Args:
        query: 查询张量
        key: 键张量
        value: 值张量
        scaleValue: 缩放因子
        sparseBlockSize: 稀疏块大小
        layoutQuery: 查询布局
    
    Returns:
        输出张量
    """

    scores = torch.matmul(query, key.transpose(-2, -1)) * scaleValue
    attn_weights = torch.nn.functional.softmax(scores, dim=-1)
    y = torch.matmul(attn_weights, value)
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import ascend_bench

B, S, N, D = 2, 1024, 8, 64
query = torch.randn(B, S, N, D, dtype=torch.float16, device="npu")
key = torch.randn(B, S, N, D, dtype=torch.float16, device="npu")
value = torch.randn(B, S, N, D, dtype=torch.float16, device="npu")
y = ascend_bench.sparse_flash_attention(query, key, value,
                                         scaleValue=1.0 / (D ** 0.5),
                                         sparseBlockSize=64,
                                         layoutQuery="BSND")
```

### 性能基线参考

当前暂无测试用例和性能基线数据。

### 相关算子

- **MlaProlog**：Multi-Head Latent Attention 前处理，作为注意力计算的前置步骤
- **GroupedMatmulSwigluQuant**：同为 L4 级融合复合算子，涉及矩阵乘法融合
- **GRU**：同为序列处理相关的 L4 级循环网络算子
