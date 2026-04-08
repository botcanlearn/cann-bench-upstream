import torch

"""
Exp算子Torch Golden参考实现

计算输入张量的指数函数，支持自定义底数、缩放和偏移
公式: y = e^((x * scale + shift) * ln(base))
"""
def exp(
    x: torch.Tensor, base: float = -1.0, scale: float = 1.0, shift: float = 0.0
) -> torch.Tensor:
    """
    计算输入张量的指数函数，支持自定义底数、缩放和偏移
    
    公式: y = e^((x * scale + shift) * ln(base))
    
    Args:
        x: 输入张量
        base: 指数底数，-1.0表示使用自然底数e，正值表示自定义底数
        scale: 输入缩放因子
        shift: 输入偏移量
    
    Returns:
        指数计算结果
    """

    temp = x * scale + shift
    if base == -1.0:
        y = torch.exp(temp)
    else:
        y = torch.pow(torch.tensor(base, dtype=x.dtype, device=x.device), temp)
    return y
