import torch

"""
GridSampler3D算子Torch Golden参考实现

根据grid中坐标信息填充输出
公式: y = grid_sample(x, grid)
"""
def grid_sampler3_d(
    x: torch.Tensor, grid: torch.Tensor, interpolation_mode: str = 'bilinear', padding_mode: str = 'zeros', align_corners: bool = False
) -> torch.Tensor:
    """
    根据grid中坐标信息填充输出
    
    公式: y = grid_sample(x, grid)
    
    Args:
        x: 输入张量
        grid: 采样网格
        interpolation_mode: 插值模式 ('bilinear': 双线性, 'nearest': 最近邻, 'bicubic': 双三次)
        padding_mode: 填充模式 ('zeros': 零填充, 'border': 边界填充, 'reflection': 反射填充)
        align_corners: 是否对齐角点
    
    Returns:
        输出张量，采样结果
    """

    return torch.nn.functional.grid_sample(
        x, grid, mode=interpolation_mode, padding_mode=padding_mode, align_corners=align_corners
    )
