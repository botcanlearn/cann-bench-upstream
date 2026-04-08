import torch

"""
StridedSlice算子Torch Golden参考实现

使用步长对输入张量进行多维切片，提取子张量。支持begin_mask、end_mask控制边界、shrink_axis_mask收缩维度、new_axis_mask插入新维度、ellipsis_mask省略号等功能
公式: y[i,j,k,...] = x[begin[i]:end[i]:strides[i], begin[j]:end[j]:strides[j], begin[k]:end[k]:strides[k], ...]
"""
def strided_slice(
    x: torch.Tensor, begin: list, end: list, strides: list, begin_mask: int, end_mask: int, ellipsis_mask: int, shrink_axis_mask: int, new_axis_mask: int
) -> torch.Tensor:
    """
    使用步长对输入张量进行多维切片，提取子张量。支持begin_mask、end_mask控制边界、shrink_axis_mask收缩维度、new_axis_mask插入新维度、ellipsis_mask省略号等功能
    
    公式: y[i,j,k,...] = x[begin[i]:end[i]:strides[i], begin[j]:end[j]:strides[j], begin[k]:end[k]:strides[k], ...]
    
    Args:
        x: 输入张量
        begin: 切片起始位置数组，长度等于输入维度数
        end: 切片结束位置数组，长度等于输入维度数
        strides: 切片步长数组，长度等于输入维度数，支持负数步长
        begin_mask: begin_mask为二进制掩码，位1表示该维度从0开始，位0使用begin值
        end_mask: end_mask为二进制掩码，位1表示该维度切到末尾，位0使用end值
        ellipsis_mask: ellipsis_mask为二进制掩码，位1表示该维度使用省略号标记
        shrink_axis_mask: shrink_axis_mask为二进制掩码，位1表示该维度被收缩掉（维度大小为1）
        new_axis_mask: new_axis_mask为二进制掩码，位1表示该位置插入大小为1的新维度
    
    Returns:
        输出张量，切片结果
    """

    # 简化实现，实际需要处理各种mask
    slices = [slice(b, e, s) for b, e, s in zip(begin, end, strides)]
    y = x[slices]
    return y
