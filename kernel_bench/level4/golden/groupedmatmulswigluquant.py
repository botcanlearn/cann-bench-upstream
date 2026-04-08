import torch

"""
GroupedMatmulSwigluQuant算子Torch Golden参考实现

分组矩阵乘法与SwiGLU及量化的融合
公式: y = SwiGLU(Dequant(Matmul(x, weight)))
"""
def grouped_matmul_swiglu_quant(
    x: torch.Tensor, weight: torch.Tensor, dequantMode: int, isEnableWeightAssistanceMatrix: bool = False
) -> torch.Tensor:
    """
    分组矩阵乘法与SwiGLU及量化的融合
    
    公式: y = SwiGLU(Dequant(Matmul(x, weight)))
    
    Args:
        x: 输入矩阵
        weight: 权重矩阵
        dequantMode: 反量化模式
        isEnableWeightAssistanceMatrix: 是否启用权重辅助矩阵
    
    Returns:
        输出张量
    """

    matmul_result = torch.matmul(x.float(), weight.float())
    
    if dequantMode == 0:
        dequant_result = matmul_result * 0.1
    else:
        dequant_result = matmul_result
    
    half_dim = dequant_result.shape[-1] // 2
    x_left = dequant_result[..., :half_dim]
    x_right = dequant_result[..., half_dim:]
    result = torch.nn.functional.silu(x_left) * x_right
    
    scale = 127.0 / result.abs().max()
    y = torch.clamp((result * scale).round(), -128, 127).to(torch.int8)
    return y
