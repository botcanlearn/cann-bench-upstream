import torch

"""
MaxPool3DGradWithArgmax算子Torch Golden参考实现

将梯度回填到每个窗口最大值的坐标处
公式: y = max_pool3d_grad(dy, argmax)
"""
def max_pool3_d_grad_with_argmax(
    x: torch.Tensor, dy: torch.Tensor, argmax: torch.Tensor, ksize: list, strides: list, pads: list
) -> torch.Tensor:
    """
    将梯度回填到每个窗口最大值的坐标处
    
    公式: y = max_pool3d_grad(dy, argmax)
    
    Args:
        x: 正向传播的原始输入张量
        dy: 输入梯度张量
        argmax: 前向传播的最大值索引
        ksize: 池化核大小 [D, H, W]
        strides: 步长 [D, H, W]
        pads: 填充 [pad_front, pad_back, pad_top, pad_bottom, pad_left, pad_right]
    
    Returns:
        输出梯度张量
    """

    # MaxPool3D反向传播：将梯度回填到argmax指定的位置
    y = torch.zeros_like(x)
    
    # 简化实现：使用max_unpool3d的反向传播逻辑
    # argmax需要转换为全局索引
    batch_size, channels, d_in, h_in, w_in = x.shape
    _, _, d_out, h_out, w_out = dy.shape
    
    # 将argmax展平并回填梯度
    for b in range(batch_size):
        for c in range(channels):
            for d_idx in range(d_out):
                for h_idx in range(h_out):
                    for w_idx in range(w_out):
                        if d_idx < dy.shape[2] and h_idx < dy.shape[3] and w_idx < dy.shape[4]:
                            # 获取argmax索引并回填梯度
                            idx = argmax[b, c, d_idx, h_idx, w_idx].item()
                            if idx >= 0 and idx < d_in * h_in * w_in:
                                d_orig = idx // (h_in * w_in)
                                h_orig = (idx % (h_in * w_in)) // w_in
                                w_orig = idx % w_in
                                y[b, c, d_orig, h_orig, w_orig] = dy[b, c, d_idx, h_idx, w_idx]
    
    return y
