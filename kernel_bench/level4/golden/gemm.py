import torch

"""
Gemm算子Torch Golden参考实现

通用矩阵乘法算子
公式: y = alpha * op(a) @ op(b) + beta * c
"""
def gemm(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, alpha: float = 1.0, beta: float = 1.0, transpose_a: bool = False, transpose_b: bool = False
) -> torch.Tensor:
    """
    通用矩阵乘法算子
    
    公式: y = alpha * op(a) @ op(b) + beta * c
    
    Args:
        a: 第1个输入矩阵
        b: 第2个输入矩阵
        c: 第3个输入矩阵
        alpha: alpha系数,用于缩放op(a) @ op(b)的结果
        beta: beta系数,用于缩放输入c的值
        transpose_a: 是否转置a
        transpose_b: 是否转置b
    
    Returns:
        输出张量
    """

    if transpose_a:
        a = a.transpose(-2, -1)
    if transpose_b:
        b = b.transpose(-2, -1)
    y = alpha * torch.matmul(a, b) + beta * c
    return y
