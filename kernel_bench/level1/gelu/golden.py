import torch

"""
Gelu 算子 Torch Golden 参考实现

高斯误差线性单元激活函数
公式：y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
"""
def gelu(
    x: torch.Tensor,
    approximate: str = "none"
) -> torch.Tensor:
    """
    高斯误差线性单元激活函数

    公式：y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    Args:
        x: 输入张量
        approximate: GELU 近似计算算法，可选值：'none'(精确计算) 或 'tanh'(tanh 近似)

    Returns:
        输出张量，GELU 激活结果
    """

    y = torch.nn.functional.gelu(x, approximate=approximate)
    return y
