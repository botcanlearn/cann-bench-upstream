import torch
from typing import List

"""
ResizeBilinearV2Grad 算子 Torch Golden 参考实现

ResizeBilinearV2 的反向传播
公式：y = resize_bilinear_grad(grads, original_size)
"""
def resize_bilinear_v2_grad(
    grads: torch.Tensor,
    original_size: List[int],
    align_corners: bool = False
) -> torch.Tensor:
    """
    ResizeBilinearV2 的反向传播

    Args:
        grads: 输入梯度张量，形状为 (N, C, H_out, W_out)
        original_size: 原始尺寸 [original_height, original_width]
        align_corners: 是否对齐角点

    Returns:
        输出梯度张量，形状为 (N, C, H_in, W_in)
    """

    grads_input = torch.nn.functional.interpolate(
        grads, size=original_size, mode='bilinear', align_corners=align_corners
    )
    return grads_input
