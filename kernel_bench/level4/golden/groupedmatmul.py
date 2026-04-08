import torch

"""
GroupedMatmul算子Torch Golden参考实现

分组矩阵乘法算子
公式: y[i] = x[i] @ weight[i]
"""
def grouped_matmul(
    x: torch.Tensor, weight: torch.Tensor, split_item: int = 0, transpose_weight: bool = False
) -> torch.Tensor:
    """
    分组矩阵乘法算子
    
    公式: y[i] = x[i] @ weight[i]
    
    Args:
        x: 输入矩阵
        weight: 权重矩阵
        split_item: 分组项
        transpose_weight: 是否转置权重
    
    Returns:
        输出张量
    """

    # 分组矩阵乘法
    y = torch.matmul(x, weight.transpose(-2, -1) if transpose_weight else weight)
    return y
