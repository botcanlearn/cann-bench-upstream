import torch

"""
BatchMatMul算子Torch Golden参考实现

批量矩阵乘法算子
公式: y = x1 @ x2 + bias
"""
def batch_mat_mul(
    x1: torch.Tensor, x2: torch.Tensor, bias: torch.Tensor, adj_x1: bool = False, adj_x2: bool = False
) -> torch.Tensor:
    """
    批量矩阵乘法算子
    
    公式: y = x1 @ x2 + bias
    
    Args:
        x1: 第1个输入矩阵
        x2: 第2个输入矩阵
        bias: 偏置张量
        adj_x1: 是否转置x1 (true表示对x1进行转置操作,将[B,M,K]变为[B,K,M])
        adj_x2: 是否转置x2 (true表示对x2进行转置操作,将[B,K,N]变为[B,N,K])
    
    Returns:
        输出张量
    """

    if adj_x1:
        x1 = x1.transpose(-2, -1)
    if adj_x2:
        x2 = x2.transpose(-2, -1)
    y = torch.matmul(x1, x2)
    if bias is not None:
        y = y + bias
    return y
