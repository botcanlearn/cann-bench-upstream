import torch

"""
ROIPooling 算子 Torch Golden 参考实现

对输入特征图按ROI进行最大池化
公式: y = roi_pool(x, rois, output_size)

优先使用 torchvision.ops.roi_pool（如果环境可用），否则使用手写实现。
"""

# 尝试导入 torchvision，如果可用则使用标准 API
HAS_TORCHVISION = False
tv_ops = None
try:
    import torchvision
    from torchvision.ops import roi_pool as tv_roi_pool
    HAS_TORCHVISION = True
    tv_ops = torchvision.ops
except (ImportError, RuntimeError, Exception):
    # torchvision 可能不可用，或与当前 torch 版本不兼容
    pass


def _roi_pooling_torchvision(
    x: torch.Tensor, rois: torch.Tensor, pooled_h: int, pooled_w: int,
    spatial_scale: float = 1.0
) -> torch.Tensor:
    """
    使用 torchvision.ops.roi_pool 实现

    Args:
        x: 输入特征图 [N, C, H, W]
        rois: ROI框 [K, 5] (batch_idx, x1, y1, x2, y2)
        pooled_h: 池化后高度
        pooled_w: 池化后宽度
        spatial_scale: 空间缩放因子

    Returns:
        输出张量 [K, C, pooled_h, pooled_w]
    """
    return tv_roi_pool(
        x, rois,
        output_size=(pooled_h, pooled_w),
        spatial_scale=spatial_scale
    )


def _roi_pooling_manual(
    x: torch.Tensor, rois: torch.Tensor, pooled_h: int, pooled_w: int,
    spatial_scale: float = 1.0
) -> torch.Tensor:
    """
    手写实现：对输入特征图按ROI进行最大池化

    公式: y = roi_pool(x, rois, output_size)

    Args:
        x: 输入特征图 [N, C, H, W]
        rois: ROI框 [K, 5] (batch_idx, x1, y1, x2, y2)
        pooled_h: 池化后高度
        pooled_w: 池化后宽度
        spatial_scale: 空间缩放因子

    Returns:
        输出张量 [K, C, pooled_h, pooled_w]
    """

    num_rois = rois.shape[0]
    channels = x.shape[1]

    output = torch.zeros(
        (num_rois, channels, pooled_h, pooled_w),
        dtype=x.dtype, device=x.device
    )

    for i in range(num_rois):
        roi = rois[i]
        batch_idx = int(roi[0])
        x1, y1, x2, y2 = roi[1:].tolist()

        x1 *= spatial_scale
        y1 *= spatial_scale
        x2 *= spatial_scale
        y2 *= spatial_scale

        roi_width = max(x2 - x1, 0.0)
        roi_height = max(y2 - y1, 0.0)

        bin_size_h = roi_height / pooled_h
        bin_size_w = roi_width / pooled_w

        roi_data = x[batch_idx:batch_idx + 1]

        for ph in range(pooled_h):
            for pw in range(pooled_w):
                y_start = int(y1 + ph * bin_size_h)
                x_start = int(x1 + pw * bin_size_w)
                y_end = int(y1 + (ph + 1) * bin_size_h)
                x_end = int(x1 + (pw + 1) * bin_size_w)

                y_start = max(y_start, 0)
                x_start = max(x_start, 0)
                y_end = min(y_end, roi_data.shape[2])
                x_end = min(x_end, roi_data.shape[3])

                if y_end > y_start and x_end > x_start:
                    output[i, :, ph, pw] = roi_data[:, :, y_start:y_end, x_start:x_end].max(dim=2)[0].max(dim=2)[0]

    return output


def roi_pooling(
    x: torch.Tensor, rois: torch.Tensor, pooled_h: int, pooled_w: int,
    spatial_scale: float = 1.0
) -> torch.Tensor:
    """
    对输入特征图按ROI进行最大池化

    公式: y = roi_pool(x, rois, output_size)

    如果 torchvision 可用，使用 torchvision.ops.roi_pool；
    否则使用手写实现。

    Args:
        x: 输入特征图 [N, C, H, W]
        rois: ROI框 [K, 5] (batch_idx, x1, y1, x2, y2)
        pooled_h: 池化后高度
        pooled_w: 池化后宽度
        spatial_scale: 空间缩放因子

    Returns:
        输出张量 [K, C, pooled_h, pooled_w]
    """
    if HAS_TORCHVISION:
        return _roi_pooling_torchvision(x, rois, pooled_h, pooled_w, spatial_scale)
    else:
        return _roi_pooling_manual(x, rois, pooled_h, pooled_w, spatial_scale)
