import torch

"""
ReduceSum算子Torch Golden参考实现

返回给定维度中输入张量每行的和
公式: y = sum(x, axis=axes, keepdims=keep_dims)
"""
def reduce_sum(
    x: torch.Tensor, axes: list, keep_dims: bool = False
) -> torch.Tensor:
    """
    返回给定维度中输入张量每行的和
    
    公式: y = sum(x, axis=axes, keepdims=keep_dims)
    
    Args:
        x: 输入张量
        axes: 归约的维度
        keep_dims: 是否保持维度
    
    Returns:
        输出张量，求和结果
    """

    y = torch.sum(x, dim=axes, keepdim=keep_dims)
    return y
