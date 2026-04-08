import torch

"""
DynamicQuant算子Torch Golden参考实现

对输入张量进行per-token对称动态量化
公式: scaleOut = row_max(abs(x)) / dtypeMax, yOut = round(x / scaleOut)
"""
def dynamic_quant(
    x: torch.Tensor, axis: int = -1, dst_type: int = 0
) -> torch.Tensor:
    """
    对输入张量进行per-token对称动态量化
    
    公式: scaleOut = row_max(abs(x)) / dtypeMax, yOut = round(x / scaleOut)
    
    Args:
        x: 输入张量
        axis: 计算scale和zero_point的维度，默认为最后一个维度
        dst_type: 目标数据类型
    
    Returns:
        量化后的张量
    """

    scale_out = torch.max(torch.abs(x), dim=axis, keepdim=True)[0] / 127.0
    y = torch.round(x / scale_out)
    return y
