import torch

"""
Index算子Torch Golden参考实现

根据索引indices将输入x对应坐标的数据取出
公式: y = x[indices[0], indices[1], ...]
"""
def index(
    x: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    """
    根据索引indices将输入x对应坐标的数据取出
    
    公式: y = x[indices[0], indices[1], ...]
    
    Args:
        x: 输入张量
        indices: 索引列表
    
    Returns:
        输出张量，索引结果
    """

    y = x[indices]
    return y
