import torch

"""
AdaptiveAvgPool3D算子Torch Golden参考实现

完成输入张量的3D自适应平均池化计算
公式: y = adaptive_avg_pool3d(x, output_size)
"""
def adaptive_avg_pool3_d(
    x: torch.Tensor, output_size: tuple[int, int, int]
) -> torch.Tensor:
    """
    完成输入张量的3D自适应平均池化计算

    公式: y = adaptive_avg_pool3d(x, output_size)

    Args:
        x: 输入张量，shape 为 [N, C, D, H, W]
        output_size: 输出尺寸，格式为 (output_d, output_h, output_w)

    Returns:
        输出张量，池化结果
    """

    y = torch.nn.functional.adaptive_avg_pool3d(x, output_size)
    return y
