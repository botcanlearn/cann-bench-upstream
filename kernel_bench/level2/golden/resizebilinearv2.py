import torch
from typing import List, Optional

"""
ResizeBilinearV2 算子 Torch Golden 参考实现

使用双线性插值调整图像大小
公式：y = resize_bilinear(x, size)
"""
def resize_bilinear_v2(
    x: torch.Tensor,
    output_size: Optional[List[int]] = None,
    align_corners: bool = False,
    scale_factor: Optional[List[float]] = None
) -> torch.Tensor:
    """
    使用双线性插值调整图像大小

    Args:
        x: 输入张量，形状为 (N, C, H, W)
        output_size: 输出尺寸 [output_height, output_width]
        align_corners: 是否对齐角点
        scale_factor: 缩放因子 [scale_height, scale_width]，与 output_size 互斥

    Returns:
        输出张量，调整大小后的结果
    """
    # 使用 PyTorch 的 interpolate 实现双线性插值
    y = torch.nn.functional.interpolate(
        x,
        size=output_size,
        scale_factor=scale_factor[0] if scale_factor and len(scale_factor) == 1 else scale_factor,
        mode='bilinear',
        align_corners=align_corners
    )
    return y
