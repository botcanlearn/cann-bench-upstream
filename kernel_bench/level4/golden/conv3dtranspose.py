import torch

"""
Conv3DTranspose算子Torch Golden参考实现

3D转置卷积
公式: y = conv3d_transpose(x, filter)
"""
def conv3_d_transpose(
    x: torch.Tensor, filter: torch.Tensor, strides: list, pads: list, dilations: list, groups: int, output_padding: list = [0, 0, 0, 0, 0]
) -> torch.Tensor:
    """
    3D转置卷积
    
    公式: y = conv3d_transpose(x, filter)
    
    Args:
        x: 输入特征图
        filter: 卷积核
        strides: 步长
        pads: 填充
        output_padding: 输出填充
        dilations: 膨胀率
        groups: 分组数
    
    Returns:
        输出特征图
    """

    padding = (pads[0], pads[1], pads[2], pads[3], pads[4])
    stride = (strides[0], strides[1], strides[2])
    dilation = (dilations[0], dilations[1], dilations[2])
    output_padding = (output_padding[0], output_padding[1], output_padding[2])
    
    y = torch.nn.functional.conv_transpose3d(x, filter, bias=None, stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, groups=groups)
    return y
