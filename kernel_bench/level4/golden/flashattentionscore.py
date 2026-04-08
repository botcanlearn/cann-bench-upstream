import torch

"""
FlashAttentionScore算子Torch Golden参考实现

FlashAttention算法实现self-attention
公式: y = Dropout(Softmax(Mask(scale * query * key^T))) * value
"""
def flash_attention_score(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor, headNum: int, inputLayout: str, scaleValue: float = 1.0, keepProb: float = 1.0
) -> torch.Tensor:
    """
    FlashAttention算法实现self-attention
    
    公式: y = Dropout(Softmax(Mask(scale * query * key^T))) * value
    
    Args:
        query: 查询张量
        key: 键张量
        value: 值张量
        mask: 掩码张量
        scaleValue: 缩放因子
        keepProb: keep probability
        headNum: 注意力头数
        inputLayout: 输入布局
    
    Returns:
        输出张量，注意力结果
    """

    if inputLayout == 'BNSD':
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
    elif inputLayout == 'BSDN':
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
    
    scores = torch.matmul(query, key.transpose(-2, -1)) * scaleValue
    
    if mask is not None:
        scores = scores + mask
    
    attn_weights = torch.nn.functional.softmax(scores, dim=-1)
    
    if keepProb < 1.0:
        attn_weights = torch.nn.functional.dropout(attn_weights, p=1-keepProb)
    
    y = torch.matmul(attn_weights, value)
    
    if inputLayout == 'BNSD':
        y = y.transpose(1, 2)
    elif inputLayout == 'BSDN':
        y = y.permute(0, 2, 1, 3)
    
    return y
