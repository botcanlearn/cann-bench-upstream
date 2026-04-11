import torch

"""
MaskedScale算子Torch Golden参考实现

对输入张量进行掩码缩放，支持x和mask的不同数据类型组合
公式: y = x * mask * scale
"""
def masked_scale(
    x: torch.Tensor, mask: torch.Tensor, scale: float = 1.0
) -> torch.Tensor:
    """
    对输入张量进行掩码缩放，支持x和mask的不同数据类型组合
    
    公式: y = x * mask * scale
    
    Args:
        x: 输入张量
        mask: 掩码张量
        scale: 缩放因子
    
    Returns:
        掩码缩放结果
    """

    y = x * mask * scale
    return y
