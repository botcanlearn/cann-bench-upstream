import torch

"""
QuantBatchMatmul算子Torch Golden参考实现

量化批量矩阵乘法算子
公式: y = dequant(x1 @ x2 * scale) + bias
"""
def quant_batch_matmul(
    x1: torch.Tensor, x2: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor, dtype: int, transpose_x1: bool = False, transpose_x2: bool = False
) -> torch.Tensor:
    """
    量化批量矩阵乘法算子
    
    公式: y = dequant(x1 @ x2 * scale) + bias
    
    Args:
        x1: 第1个输入矩阵
        x2: 第2个输入矩阵
        scale: 量化缩放因子
        bias: 偏置张量
        dtype: 量化数据类型
        transpose_x1: 是否转置x1
        transpose_x2: 是否转置x2
    
    Returns:
        输出张量
    """

    x1_adj = x1.transpose(-2, -1) if transpose_x1 else x1
    x2_adj = x2.transpose(-2, -1) if transpose_x2 else x2
    
    matmul_result = torch.matmul(x1_adj.float(), x2_adj.float())
    scaled_result = matmul_result * scale.float()
    y = scaled_result + bias.float()
    
    return y.to(x1.dtype)
