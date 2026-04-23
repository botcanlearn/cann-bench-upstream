import torch

"""
DepthwiseConv2D算子Torch Golden参考实现

二维深度卷积运算
公式: y = bias + weight * x
"""
def depthwise_conv2_d(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, kernelSize: list, stride: list, padding: list, dilation: list, groups: int
) -> torch.Tensor:
    """
    二维深度卷积运算
    
    公式: y = bias + weight * x
    
    Args:
        x: 输入特征图
        weight: 卷积核
        bias: 偏置
        kernelSize: 卷积核大小
        stride: 步长
        padding: 填充
        dilation: 膨胀率
        groups: 分组数
    
    Returns:
        输出特征图
    """

    stride_val = (stride[0], stride[1])
    padding_val = (padding[0], padding[1])
    dilation_val = (dilation[0], dilation[1])
    
    y = torch.nn.functional.conv2d(x, weight, bias, stride=stride_val, padding=padding_val, dilation=dilation_val, groups=groups)
    return y
