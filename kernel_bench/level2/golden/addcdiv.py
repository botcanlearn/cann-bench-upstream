import torch

"""
Addcdiv算子Torch Golden参考实现

执行加法和除法组合运算，输出等于第一个输入加上第二个和第三个输入商的value倍
公式: y = x1 + value * (x2 / x3)
"""
def addcdiv(
    x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, value: float = 1.0
) -> torch.Tensor:
    """
    执行加法和除法组合运算，输出等于第一个输入加上第二个和第三个输入商的value倍
    
    公式: y = x1 + value * (x2 / x3)
    
    Args:
        x1: 第1个输入张量
        x2: 第2个输入张量
        x3: 第3个输入张量（除数）
        value: 缩放因子，乘以 x2/x3 的结果
    
    Returns:
        输出张量
    """

    y = x1 + value * (x2 / x3)
    return y
