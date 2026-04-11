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
