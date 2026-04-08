import torch

"""
Cummin算子Torch Golden参考实现

计算输入张量中的累积最小值
公式: y[i] = min(x[0], x[1], ..., x[i]) 沿指定轴
"""
def cummin(
    x: torch.Tensor, axis: int
) -> torch.Tensor:
    """
    计算输入张量中的累积最小值
    
    公式: y[i] = min(x[0], x[1], ..., x[i]) 沿指定轴
    
    Args:
        x: 输入张量
        axis: 计算累积最小值的轴
    
    Returns:
        输出张量，累积最小值
    """

    y = torch.cummin(x, dim=axis)[0]
    return y
