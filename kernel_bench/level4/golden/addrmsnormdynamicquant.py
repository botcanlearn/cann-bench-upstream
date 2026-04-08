import torch

"""
AddRmsNormDynamicQuant 算子 Torch Golden 参考实现

Add、RMSNorm 和动态量化的融合
公式：y, xOut, scaleOut = quantize(rmsnorm(x1 + x2) * gamma)
"""
def add_rms_norm_dynamic_quant(
    x1: torch.Tensor,
    x2: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float = 1e-6,
    dst_type: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Add、RMSNorm 和动态量化的融合

    公式：y, xOut, scaleOut = quantize(rmsnorm(x1 + x2) * gamma)

    Args:
        x1: 第 1 个输入张量
        x2: 第 2 个输入张量
        gamma: 缩放参数
        epsilon: epsilon 值
        dst_type: 目标数据类型 (0:DT_INT8, 1:DT_INT4, 2:DT_FP8)

    Returns:
        y: 量化后的输出张量
        xOut: Add 结果，x1 + x2
        scaleOut: 量化使用的 scale 值
    """

    # Add 操作
    xOut = x1 + x2

    # RMSNorm
    variance = xOut.pow(2).mean(-1, keepdim=True)
    rms = torch.sqrt(variance + epsilon)
    normalized = xOut / rms
    y_norm = normalized * gamma

    # 动态量化
    if dst_type == 0:  # INT8
        scale = 127.0 / y_norm.abs().max()
        y = torch.clamp((y_norm * scale).round(), -128, 127).to(torch.int8)
    elif dst_type == 1:  # INT4
        scale = 7.0 / y_norm.abs().max()
        y = torch.clamp((y_norm * scale).round(), -8, 7).to(torch.int8)
    else:  # FP8
        scale = 240.0 / y_norm.abs().max()
        y = torch.clamp((y_norm * scale).round(), -128, 127).to(torch.float8_e4m3fn)

    return y, xOut, scale
