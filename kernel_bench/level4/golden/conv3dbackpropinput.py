import torch

"""
Conv3DBackpropInput算子Torch Golden参考实现

Conv3D的输入梯度
公式: y = conv3d_input_grad(filter, grad)
"""
def conv3_d_backprop_input(
    filter: torch.Tensor, grad: torch.Tensor, strides: list, pads: list, dilations: list
) -> torch.Tensor:
    """
    Conv3D的输入梯度
    
    公式: y = conv3d_input_grad(filter, grad)
    
    Args:
        filter: 卷积核
        grad: 输出梯度
        strides: 步长
        pads: 填充
        dilations: 膨胀率
    
    Returns:
        输入梯度
    """

    padding = (pads[0], pads[1], pads[2], pads[3], pads[4])
    stride = (strides[0], strides[1], strides[2])
    dilation = (dilations[0], dilations[1], dilations[2])
    
    y = torch.nn.functional.conv_transpose3d(grad, filter, bias=None, stride=stride, padding=padding, dilation=dilation)
    return y
