import torch

"""
Add算子Torch Golden参考实现

对两个输入张量完成相加操作，支持广播
公式: y = x1 + x2
"""
def add(
    x1: torch.Tensor, x2: torch.Tensor
) -> torch.Tensor:
    """
    对两个输入张量完成相加操作，支持广播
    
    公式: y = x1 + x2
    
    Args:
        x1: 第1个输入张量
        x2: 第2个输入张量
    
    Returns:
        输出张量
    """

    y = x1 + x2
    return y
