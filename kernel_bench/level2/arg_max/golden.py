import torch

"""
ArgMax算子Torch Golden参考实现

返回张量在指定维度上的最大值的索引
公式: y = argmax(x, axis=dimension)
"""
def arg_max(
    x: torch.Tensor, dimension: int
) -> torch.Tensor:
    """
    返回张量在指定维度上的最大值的索引
    
    公式: y = argmax(x, axis=dimension)
    
    Args:
        x: 输入张量
        dimension: 计算argmax的维度
    
    Returns:
        输出张量，最大值的索引
    """

    y = torch.argmax(x, dim=dimension)
    return y
