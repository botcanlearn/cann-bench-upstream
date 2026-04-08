import torch

"""
MoeGatingTopKSoftmax算子Torch Golden参考实现

MoE门控网络中Softmax和TopK的融合
公式: y = TopK(Softmax(x), k)
"""
def moe_gating_top_k_softmax(
    x: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    MoE门控网络中Softmax和TopK的融合
    
    公式: y = TopK(Softmax(x), k)
    
    Args:
        x: 输入张量
        k: topK数量
    
    Returns:
        y, softmax_out
    """

    softmax_out = torch.nn.functional.softmax(x, dim=-1)
    values, indices = torch.topk(softmax_out, k, dim=-1)
    return values, softmax_out
