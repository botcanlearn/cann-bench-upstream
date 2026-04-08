import torch

"""
ROIPooling算子Torch Golden参考实现

对输入特征图按ROI进行最大池化
公式: y = roi_pool(x, rois, output_size)
"""
def roi_pooling(
    x: torch.Tensor, rois: torch.Tensor, pooled_h: int, pooled_w: int, spatial_scale: float
) -> torch.Tensor:
    """
    对输入特征图按ROI进行最大池化
    
    公式: y = roi_pool(x, rois, output_size)
    
    Args:
        x: 输入特征图
        rois: ROI框
        pooled_h: 池化后高度
        pooled_w: 池化后宽度
        spatial_scale: 空间缩放因子 (用于将ROI坐标映射到输入特征图尺寸)
    
    Returns:
        输出张量，ROI池化结果
    """

    from torchvision.ops import roi_pool as tv_roi_pool
    
    output_size = (pooled_h, pooled_w)
    
    y = tv_roi_pool(
        x, rois, output_size,
        spatial_scale=spatial_scale
    )
    
    return y
