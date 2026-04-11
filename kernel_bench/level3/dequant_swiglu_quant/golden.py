import torch

"""
DequantSwigluQuant算子Torch Golden参考实现

反量化、SwiGLU和量化的融合
公式: y = quantize(SwiGLU(dequantize(x)))
"""
def dequant_swiglu_quant(
    x: torch.Tensor, activate_left: bool = False, quant_mode: str = 'static', dst_type: int = 0
) -> torch.Tensor:
    """
    反量化、SwiGLU和量化的融合
    
    公式: y = quantize(SwiGLU(dequantize(x)))
    
    Args:
        x: 输入张量
        activate_left: 是否激活左侧
        quant_mode: 量化模式'
        dst_type: 目标数据类型 (0:DT_INT8, 1:DT_INT4, 2:DT_FP8)'
    
    Returns:
        输出张量
    """

    def swiglu(x, activate_left=False):
        if activate_left:
            x_left = x[..., :x.shape[-1]//2]
            x_right = x[..., x.shape[-1]//2:]
            return x_left * torch.nn.functional.silu(x_right)
        else:
            x_left = x[..., :x.shape[-1]//2]
            x_right = x[..., x.shape[-1]//2:]
            return torch.nn.functional.silu(x_left) * x_right
    
    if x.dtype in [torch.int8, torch.int32]:
        scale = 0.1
        x_float = x.float() * scale
    else:
        x_float = x
    
    result = swiglu(x_float, activate_left)
    
    if dst_type == 0:
        scale = 127.0 / result.abs().max()
        y = torch.clamp((result * scale).round(), -128, 127).to(torch.int8)
    elif dst_type == 1:
        scale = 7.0 / result.abs().max()
        y = torch.clamp((result * scale).round(), -8, 7).to(torch.int8)
    else:
        scale = 240.0 / result.abs().max()
        y = torch.clamp((result * scale).round(), -128, 127).to(torch.float8_e4m3fn)
    
    return y
