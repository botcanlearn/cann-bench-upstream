import torch

"""
PFA算子Torch Golden参考实现

全量推理场景的FlashAttention算子
公式: y = Attention(Q,K,V) = Softmax(QK^T / sqrt(d)) @ V
"""
def pfa(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, numHeads: int, scaleValue: float, inputLayout: str
) -> torch.Tensor:
    """
    全量推理场景的FlashAttention算子
    
    公式: y = Attention(Q,K,V) = Softmax(QK^T / sqrt(d)) @ V
    
    Args:
        query: 查询张量
        key: 键张量
        value: 值张量
        numHeads: 注意力头数
        scaleValue: 缩放因子
        inputLayout: 输入布局
    
    Returns:
        输出张量
    """

    scores = torch.matmul(query, key.transpose(-2, -1)) * scaleValue
    attn_weights = torch.nn.functional.softmax(scores, dim=-1)
    y = torch.matmul(attn_weights, value)
    return y
