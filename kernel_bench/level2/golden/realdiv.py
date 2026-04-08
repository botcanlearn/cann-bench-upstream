import torch

"""
RealDiv算子Torch Golden参考实现

实数除法运算，支持广播
公式: y = x1 / x2
"""
def real_div(
    x1: torch.Tensor, x2: torch.Tensor
) -> torch.Tensor:
    """
    实数除法运算，支持广播
    
    公式: y = x1 / x2
    
    Args:
        x1: 第1个输入张量（被除数）
        x2: 第2个输入张量（除数）
    
    Returns:
        输出张量
    """

    y = x1 / x2
    return y
