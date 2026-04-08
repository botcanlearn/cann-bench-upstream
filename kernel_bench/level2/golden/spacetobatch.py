import torch

"""
SpaceToBatch算子Torch Golden参考实现

将空间维度的数据重排到batch维度，用于空洞卷积等场景
公式: y = space_to_batch(x, block_shape, paddings)
"""
def space_to_batch(
    x: torch.Tensor, block_shape: list, paddings: list
) -> torch.Tensor:
    """
    将空间维度的数据重排到batch维度，用于空洞卷积等场景
    
    公式: y = space_to_batch(x, block_shape, paddings)
    
    Args:
        x: 输入张量
        block_shape: 空间维度的块大小 [block_h, block_w]
        paddings: 空间维度的填充 [[pad_top, pad_bottom], [pad_left, pad_right]]
    
    Returns:
        输出张量，空间维度重排后的结果
    """

    # 4D: [N, H, W, C] -> [N*block_h*block_w, H/block_h, W/block_w, C]
    N, H, W, C = x.shape
    block_h, block_w = block_shape
    y = x.reshape(N, block_h, H // block_h, block_w, W // block_w, C)
    y = y.permute(0, 2, 4, 1, 3, 5)
    y = y.reshape(N * block_h * block_w, H // block_h, W // block_w, C)
    return y
