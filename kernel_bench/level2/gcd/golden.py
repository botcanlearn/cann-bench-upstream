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
        输出张量，最大公约数
    """

    # 转换为int64
    x1_int = x1.to(torch.int64)
    x2_int = x2.to(torch.int64)

    # torch.gcd不支持自动broadcast，需要手动处理
    # 先进行broadcast，再计算gcd
    x1_broadcast, x2_broadcast = torch.broadcast_tensors(x1_int, x2_int)

    y = torch.gcd(x1_broadcast, x2_broadcast)
    return y