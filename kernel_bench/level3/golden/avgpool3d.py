import torch

"""
AvgPool3D算子Torch Golden参考实现

对5D输入张量(N,C,D,H,W)进行三维平均池化，在深度(D)、高度(H)、宽度(W)三个维度上以指定核大小进行滑动窗口平均计算
公式: y[n,c,d,h,w] = (1/(kD*kH*kW)) * sum_{i,j,k} x[n,c,d*strideD+i,h*strideH+j,w*strideW+k]
"""
def avg_pool3_d(
    x: torch.Tensor, ksize: list, strides: list, pads: list, data_format: str = 'NCDHW', count_include_pad: bool = True
) -> torch.Tensor:
    """
    对5D输入张量(N,C,D,H,W)进行三维平均池化，在深度(D)、高度(H)、宽度(W)三个维度上以指定核大小进行滑动窗口平均计算
    
    公式: y[n,c,d,h,w] = (1/(kD*kH*kW)) * sum_{i,j,k} x[n,c,d*strideD+i,h*strideH+j,w*strideW+k]
    
    Args:
        x: 输入张量，5D张量(N,C,D,H,W)
        ksize: 池化核大小数组 [kD, kH, kW]
        strides: 步长数组 [strideD, strideH, strideW]
        pads: 填充数组 [pad_front, pad_back, pad_top, pad_bottom, pad_left, pad_right]
        data_format: 数据格式，支持 NCDHW(默认) 或 NDHWC
        count_include_pad: 是否在平均计算中包含填充值，true包含padding参与平均计算，false则padding区域不计入
    
    Returns:
        输出张量，池化后的结果
    """

    y = torch.nn.functional.avg_pool3d(x, kernel_size=ksize, stride=strides, padding=pads)
    return y
