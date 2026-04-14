import torch

"""
ROIAlign 算子 Torch Golden 参考实现

池化层，用于非均匀输入尺寸的特征图
公式: y = roi_align(x, rois, output_size)
"""
def roi_align(
    x: torch.Tensor, rois: torch.Tensor, pooled_height: int, pooled_width: int,
    spatial_scale: float = 1.0, sample_ratio: int = -1, aligned: bool = False
) -> torch.Tensor:
    """
    池化层，用于非均匀输入尺寸的特征图

    公式: y = roi_align(x, rois, output_size)

    Args:
        x: 输入特征图 [N, C, H, W]
        rois: ROI框 [K, 5] (batch_idx, x1, y1, x2, y2)
        pooled_height: 输出高度
        pooled_width: 输出宽度
        spatial_scale: 空间缩放因子
        sample_ratio: 采样比率
        aligned: 是否对齐

    Returns:
        输出张量 [K, C, pooled_height, pooled_width]
    """

    num_rois = rois.shape[0]
    channels = x.shape[1]

    output = torch.zeros(
        (num_rois, channels, pooled_height, pooled_width),
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

        bin_size_h = roi_height / pooled_height
        bin_size_w = roi_width / pooled_width

        if sample_ratio <= 0:
            n_sample = max(int(max(roi_width, roi_height) // pooled_height), 1)
        else:
            n_sample = sample_ratio

        roi_data = x[batch_idx:batch_idx + 1]

        for ph in range(pooled_height):
            for pw in range(pooled_width):
                y_start = y1 + ph * bin_size_h
                x_start = x1 + pw * bin_size_w
                y_end = y_start + bin_size_h
                x_end = x_start + bin_size_w

                ys = torch.linspace(y_start, y_end, n_sample, device=x.device)
                xs = torch.linspace(x_start, x_end, n_sample, device=x.device)
                grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')

                # Normalize to [-1, 1]
                H, W = roi_data.shape[2], roi_data.shape[3]
                norm_x = 2.0 * grid_x / (W - 1) - 1.0
                norm_y = 2.0 * grid_y / (H - 1) - 1.0
                grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)

                sampled = torch.nn.functional.grid_sample(
                    roi_data, grid, mode='bilinear', padding_mode='zeros', align_corners=not aligned
                )
                output[i, :, ph, pw] = sampled.mean(dim=[0, 2, 3])

    return output
