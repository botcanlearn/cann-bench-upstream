import torch

"""
Log算子Torch Golden参考实现

对输入张量进行对数计算，支持自定义底数、缩放和偏移
公式: y = ln(x * scale + shift) / ln(base)
"""
def log(
    x: torch.Tensor, base: float = -1.0, scale: float = 1.0, shift: float = 0.0
) -> torch.Tensor:
    """
    对输入张量进行对数计算，支持自定义底数、缩放和偏移
    
    公式: y = ln(x * scale + shift) / ln(base)
    
    Args:
        x: 输入张量
        base: 对数的底数，-1.0表示使用自然底数e，正值表示自定义底数
        scale: 输入缩放因子
        shift: 输入偏移量
    
    Returns:
        对数计算结果
    """

    temp = x * scale + shift
    if base == -1.0:
        y = torch.log(temp)
    else:
        y = torch.log(temp) / torch.log(torch.tensor(base, dtype=x.dtype, device=x.device))
    return y
