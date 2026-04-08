import torch

"""
Maximum算子Torch Golden参考实现

返回两个输入张量中的最大值，支持广播
公式: y = max(x1, x2)
"""
def maximum(
    x1: torch.Tensor, x2: torch.Tensor
) -> torch.Tensor:
    """
    返回两个输入张量中的最大值，支持广播
    
    公式: y = max(x1, x2)
    
    Args:
        x1: 第1个输入张量
        x2: 第2个输入张量
    
    Returns:
        输出张量，两个输入中的最大值
    """

    y = torch.maximum(x1, x2)
    return y
