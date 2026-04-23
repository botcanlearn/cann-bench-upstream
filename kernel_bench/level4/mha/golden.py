import torch

"""
MHA算子Torch Golden参考实现

多头注意力 (Multi-Head Attention)，对已分头的 Q/K/V 执行缩放点积注意力
公式: y = softmax(Q @ K^T * scaleValue) @ V
"""


def mha(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaleValue: float = -1.0,
) -> torch.Tensor:
    """
    多头注意力 (Multi-Head Attention)

    Args:
        query: 查询张量 [B, S, N, D]（已分头）
        key: 键张量 [B, S_kv, N, D]（已分头）
        value: 值张量 [B, S_kv, N, D]（已分头）
        scaleValue: 缩放因子，<=0 时自动使用 1/sqrt(D)

    Returns:
        输出张量 [B, S, N, D]
    """
    B, S, N, D = query.shape

    if scaleValue <= 0:
        scaleValue = 1.0 / (D ** 0.5)

    # 转置为 [B, N, S, D]
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)

    # 缩放点积注意力
    scores = torch.matmul(q, k.transpose(-2, -1)) * scaleValue
    attn_weights = torch.nn.functional.softmax(scores, dim=-1)
    attn_output = torch.matmul(attn_weights, v)

    # 转回 [B, S, N, D]
    return attn_output.transpose(1, 2)
