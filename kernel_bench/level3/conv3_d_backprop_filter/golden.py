import torch

"""
Conv3DBackpropFilter算子Torch Golden参考实现

Conv3D的filter梯度
公式: y = conv3d_filter_grad(x, grad)
"""
def conv3_d_backprop_filter(
    x: torch.Tensor, grad: torch.Tensor, strides: list, pads: list, dilations: list
) -> torch.Tensor:
    """
    Conv3D的filter梯度
    
    公式: y = conv3d_filter_grad(x, grad)
    
    Args:
        x: 输入特征图
        grad: 输出梯度
        strides: 步长
        pads: 填充
        dilations: 膨胀率
    
    Returns:
        filter梯度
    """

    padding = (pads[0], pads[1], pads[2], pads[3], pads[4])
    stride = (strides[0], strides[1], strides[2])
    dilation = (dilations[0], dilations[1], dilations[2])
    
    y = torch.nn.functional.conv3d(x, grad, bias=None, stride=stride, padding=padding, dilation=dilation)
    return y
