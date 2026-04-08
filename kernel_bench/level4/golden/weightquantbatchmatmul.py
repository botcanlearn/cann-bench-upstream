import torch

"""
WeightQuantBatchMatmul算子Torch Golden参考实现

权重量化批量矩阵乘法算子
公式: y = quant(dequant(weight) @ x + bias)
"""
def weight_quant_batch_matmul(
    weight: torch.Tensor, x: torch.Tensor, bias: torch.Tensor, transpose_x: bool = False, transpose_weight: bool = False
) -> torch.Tensor:
    """
    权重量化批量矩阵乘法算子
    
    公式: y = quant(dequant(weight) @ x + bias)
    
    Args:
        weight: 权重矩阵
        x: 输入矩阵
        bias: 偏置张量
        transpose_x: 是否转置x
        transpose_weight: 是否转置权重
    
    Returns:
        输出张量
    """

    weight_adj = weight.transpose(-2, -1) if transpose_weight else weight
    x_adj = x.transpose(-2, -1) if transpose_x else x
    
    if weight.dtype in [torch.int8, torch.int4]:
        weight_float = weight.float() * 0.1
    else:
        weight_float = weight.float()
    
    matmul_result = torch.matmul(weight_float, x_adj.float())
    result = matmul_result + bias.float()
    
    scale = 127.0 / result.abs().max()
    y = torch.clamp((result * scale).round(), -128, 127).to(weight.dtype)
    return y
