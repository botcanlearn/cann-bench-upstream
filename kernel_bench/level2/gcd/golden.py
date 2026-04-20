import torch

"""
Gcd算子Torch Golden参考实现

计算两个整数的最大公约数
公式: y = gcd(x1, x2)
"""
def gcd(
    x1: torch.Tensor, x2: torch.Tensor
) -> torch.Tensor:
    """
    计算两个整数的最大公约数

    公式: y = gcd(x1, x2)

    Args:
        x1: 第1个输入张量
        x2: 第2个输入张量

    Returns:
        输出张量，最大公约数（dtype 与输入一致）
    """

    y = torch.gcd(x1, x2)
    return y