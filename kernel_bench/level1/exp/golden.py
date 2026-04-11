import torch

"""
Exp算子Torch Golden参考实现

计算输入张量的指数函数
- base <= 0: y = exp(scale * x + shift)
- base > 0: y = exp((shift + scale * x) * ln(base))
"""
def exp(
    x: torch.Tensor, base: float = -1.0, scale: float = 1.0, shift: float = 0.0
) -> torch.Tensor:
    """
    计算输入张量的指数函数

    - base <= 0: y = exp(scale * x + shift)
    - base > 0: y = exp((shift + scale * x) * ln(base))

    Args:
        x: 输入张量
        base: 指数底数，base <= 0 表示使用自然底数 e
        scale: 输入缩放因子
        shift: 输入偏移量

    Returns:
        指数计算结果
    """
    temp = scale * x + shift
    if base > 0:
        temp = temp * torch.log(torch.tensor(base, dtype=x.dtype, device=x.device))
    return torch.exp(temp)