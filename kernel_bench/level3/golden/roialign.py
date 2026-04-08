import torch

"""
ROIAlign算子Torch Golden参考实现

池化层，用于非均匀输入尺寸的特征图
公式: y = roi_align(x, rois, output_size)
"""
def roi_align(
    x: torch.Tensor, rois: torch.Tensor, mode: str, outputHeight: int, outputWidth: int, spatial_scale: float, sampling_ratio: int = -1, aligned: bool = False
) -> torch.Tensor:
    """
    池化层，用于非均匀输入尺寸的特征图
    
    公式: y = roi_align(x, rois, output_size)
    
    Args:
        x: 输入特征图
        rois: ROI框
        mode: 插值模式 ('bilinear': 双线性, 'nearest': 最近邻)
        outputHeight: 输出高度
        outputWidth: 输出宽度
        spatial_scale: 空间缩放因子 (用于将ROI坐标映射到输入特征图尺寸)
        sampling_ratio: 采样比率
        aligned: 是否对齐
    
    Returns:
        输出张量，ROI对齐结果
    """

    from torchvision.ops import roi_align as tv_roi_align
    
    output_size = (outputHeight, outputWidth)
    
    y = tv_roi_align(
        x, rois, output_size,
        spatial_scale=spatial_scale,
        sampling_ratio=sampling_ratio,
        aligned=aligned
    )
    
    return y
