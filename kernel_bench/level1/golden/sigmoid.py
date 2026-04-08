import torch

"""
Sigmoid算子Torch Golden参考实现

对输入Tensor完成Sigmoid运算
公式: y = 1 / (1 + e^(-x))
"""
def sigmoid(
    x: torch.Tensor
) -> torch.Tensor:
    """
    对输入Tensor完成Sigmoid运算
    
    公式: y = 1 / (1 + e^(-x))
    
    Args:
        x: 输入张量
    
    Returns:
        输出张量，Sigmoid激活结果
    """

    y = torch.sigmoid(x)
    return y
