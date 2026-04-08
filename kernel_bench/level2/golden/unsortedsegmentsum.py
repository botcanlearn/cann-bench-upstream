import torch

"""
UnsortedSegmentSum算子Torch Golden参考实现

沿segment_ids指定的段对数据进行求和
公式: y[i] = sum(data[j]) where segment_ids[j] == i
"""
def unsorted_segment_sum(
    data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """
    沿segment_ids指定的段对数据进行求和
    
    公式: y[i] = sum(data[j]) where segment_ids[j] == i
    
    Args:
        data: 输入数据张量
        segment_ids: 段ID张量
        num_segments: 段数量
    
    Returns:
        输出张量，段求和结果
    """

    y = torch.zeros(num_segments, *data.shape[1:], dtype=data.dtype, device=data.device)
    for i in range(num_segments):
        mask = (segment_ids == i)
        y[i] = data[mask].sum(dim=0)
    return y
