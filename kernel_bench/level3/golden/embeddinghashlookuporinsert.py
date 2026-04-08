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
