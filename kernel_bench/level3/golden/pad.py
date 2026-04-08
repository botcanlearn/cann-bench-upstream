import torch

"""
Pad算子Torch Golden参考实现

对输入tensor做填充
公式: y = pad(x, paddings)
"""
def pad(
    x: torch.Tensor, paddings: list
) -> torch.Tensor:
    """
    对输入tensor做填充
    
    公式: y = pad(x, paddings)
    
    Args:
        x: 输入张量
        paddings: 填充配置，按从后往前的顺序指定每维的(left, right)，如4D输入为(left, right, top, bottom)，5D输入为(left, right, top, bottom, front, back)
    
    Returns:
        输出张量，填充后的结果
    """

    y = torch.nn.functional.pad(x, paddings)
    return y
