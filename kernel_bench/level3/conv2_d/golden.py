import torch

"""
Conv2D算子Torch Golden参考实现

计算2D卷积
公式: y = CONV(x, filter) + bias
"""
def conv2_d(
    x: torch.Tensor, filter: torch.Tensor, bias: torch.Tensor, strides: list, pads: list, dilations: list = [1, 1, 1, 1], groups: Int = 1
) -> torch.Tensor:
    """
    计算2D卷积
    
    公式: y = CONV(x, filter) + bias
    
    Args:
        x: 输入特征图
        filter: 卷积核
        bias: 偏置
        strides: 步长
        pads: 填充
        dilations: 膨胀率
        groups: 分组数
    
    Returns:
        输出特征图
    """

    padding = (pads[0], pads[1], pads[2], pads[3])
    stride = (strides[0], strides[1])
    dilation = (dilations[0], dilations[1])
    
    y = torch.nn.functional.conv2d(x, filter, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    return y
