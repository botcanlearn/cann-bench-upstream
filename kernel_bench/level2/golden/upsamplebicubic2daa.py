import torch

"""
UpsampleBicubic2dAA算子Torch Golden参考实现

应用双三次抗锯齿算法进行上采样
公式: y = bicubic_upsample(x, output_size)
"""
def upsample_bicubic2d_aa(
    x: torch.Tensor, output_size: int, align_corners: bool = False
) -> torch.Tensor:
    """
    应用双三次抗锯齿算法进行上采样

    公式: y = bicubic_upsample(x, output_size)

    Args:
        x: 输入张量
        output_size: 输出尺寸
        align_corners: 是否对齐角点

    Returns:
        输出张量，上采样结果
    """
    original_dtype = x.dtype

    # PyTorch的bicubic interpolate with antialias不支持float16/bfloat16
    # 需要先转换为float32
    if original_dtype in (torch.float16, torch.bfloat16):
        x = x.float()

    output_size_tuple = (output_size, output_size)
    result = torch.nn.functional.interpolate(
        x, size=output_size_tuple, mode='bicubic', align_corners=align_corners, antialias=True
    )

    # 转换回原始dtype
    if original_dtype in (torch.float16, torch.bfloat16):
        result = result.to(original_dtype)

    return result