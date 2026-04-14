import torch

"""
Dilation2D 算子 Torch Golden 参考实现

2D形态学膨胀操作，使用最大池化在局部邻域内获取最大值
公式: y[b, y, x, c] = max_{dy,dx} x[b, y + rates[1]*dy, x + rates[2]*dx, c] * filter[dy, dx, c]
"""
def dilation2_d(
    x: torch.Tensor, kernel_size: list, strides: list,
    pads: list = [0, 0, 0, 0], dilations: list = [1, 1],
    padding_mode: str = 'SAME', ceil_mode: bool = False,
    data_format: str = 'NHWC'
) -> torch.Tensor:
    """
    2D形态学膨胀操作，使用最大池化在局部邻域内获取最大值

    公式: y[b, y, x, c] = max_{dy,dx} x[b, y + rates[1]*dy, x + rates[2]*dx, c] * filter[dy, dx, c]

    Args:
        x: 输入图像
        kernel_size: 卷积核尺寸 [height, width]
        strides: 步长 [stride_h, stride_w]
        pads: 填充值 [pad_top, pad_bottom, pad_left, pad_right]
        dilations: 膨胀率 [dilation_h, dilation_w]
        padding_mode: 填充模式：'SAME' 或 'VALID'
        ceil_mode: 是否向上取整计算输出尺寸
        data_format: 数据格式，如 'NHWC'

    Returns:
        膨胀后的图像
    """

    if data_format == 'NHWC':
        x = x.permute(0, 3, 1, 2)

    batch, channels, in_h, in_w = x.shape
    filter_h, filter_w = kernel_size[0], kernel_size[1]
    stride_h, stride_w = strides[0], strides[1]
    rate_h, rate_w = dilations[0], dilations[1]

    effective_filter_h = (filter_h - 1) * rate_h + 1
    effective_filter_w = (filter_w - 1) * rate_w + 1

    if padding_mode == 'SAME':
        out_h = (in_h + stride_h - 1) // stride_h
        out_w = (in_w + stride_w - 1) // stride_w
        if ceil_mode:
            out_h = (in_h + stride_h - 1) // stride_h + (1 if (in_h - 1) % stride_h else 0)
            out_w = (in_w + stride_w - 1) // stride_w + (1 if (in_w - 1) % stride_w else 0)
        pad_h = max((out_h - 1) * stride_h + effective_filter_h - in_h, 0)
        pad_w = max((out_w - 1) * stride_w + effective_filter_w - in_w, 0)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        x = torch.nn.functional.pad(x, [pad_left, pad_right, pad_top, pad_bottom])
    elif padding_mode == 'VALID':
        if pads and sum(pads) > 0:
            x = torch.nn.functional.pad(x, [pads[2], pads[3], pads[0], pads[1]])
        out_h = (in_h - effective_filter_h + stride_h) // stride_h
        out_w = (in_w - effective_filter_w + stride_w) // stride_w
    else:
        if pads and sum(pads) > 0:
            x = torch.nn.functional.pad(x, [pads[2], pads[3], pads[0], pads[1]])
        out_h = (x.shape[2] - effective_filter_h + stride_h) // stride_h
        out_w = (x.shape[3] - effective_filter_w + stride_w) // stride_w
        if ceil_mode:
            out_h = (x.shape[2] - effective_filter_h + stride_h - 1) // stride_h + 1
            out_w = (x.shape[3] - effective_filter_w + stride_w - 1) // stride_w + 1

    # 形态学膨胀: 使用 unfold 获取 patches，然后取最大值
    patches = torch.nn.functional.unfold(
        x,
        kernel_size=(effective_filter_h, effective_filter_w),
        dilation=(rate_h, rate_w),
        stride=(stride_h, stride_w)
    )

    patches = patches.view(batch, channels, filter_h, filter_w, out_h, out_w)

    # 形态学膨胀：对每个位置取 filter 全为1情况下的最大值
    y = patches.max(dim=4)[0].max(dim=3)[0]

    if data_format == 'NHWC':
        y = y.permute(0, 2, 3, 1)

    return y
