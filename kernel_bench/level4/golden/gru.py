import torch

"""
GRU 算子 Torch Golden 参考实现

Gated Recurrent Unit 循环神经网络
公式:
    z_t = σ(W_z @ x_t + U_z @ h_{t-1} + b_z)
    r_t = σ(W_r @ x_t + U_r @ h_{t-1} + b_r)
    n_t = tanh(W_n @ x_t + r_t ⊙ (U_n @ h_{t-1} + b_n))
    h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}
"""
def gru(
    x: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor | None = None,
    bias_hh: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    inputSize: int = 0,
    hiddenSize: int = 0,
    numLayers: int = 1,
    bias: bool = True,
    batchFirst: bool = False,
    dropout: float = 0.0,
    bidirectional: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gated Recurrent Unit 循环神经网络

    Args:
        x: 输入序列张量 (S, B, input_size) 或 (B, S, input_size) if batch_first
        weight_ih: 输入到隐藏层权重 (num_layers * 3 * hiddenSize, inputSize)
        weight_hh: 隐藏层到隐藏层权重 (num_layers * 3 * hiddenSize, hiddenSize)
        bias_ih: 输入到隐藏层偏置 (可选)
        bias_hh: 隐藏层到隐藏层偏置 (可选)
        h0: 初始隐藏状态 (可选)
        inputSize: 输入特征维度
        hiddenSize: 隐藏状态特征维度
        numLayers: 循环层数
        bias: 是否使用偏置
        batchFirst: 输入是否为 (batch, seq, feature) 格式
        dropout: Dropout 概率
        bidirectional: 是否双向 GRU

    Returns:
        y: 输出序列
        hn: 最终隐藏状态
    """
    num_directions = 2 if bidirectional else 1

    # 使用 torch.nn.GRU 实现
    gru_layer = torch.nn.GRU(
        input_size=inputSize,
        hidden_size=hiddenSize,
        num_layers=numLayers,
        bias=bias,
        batch_first=batchFirst,
        dropout=dropout if numLayers > 1 else 0.0,
        bidirectional=bidirectional
    )

    # 手动设置权重以匹配传入的 weight
    # PyTorch GRU 权重格式：
    # weight_ih_l[k]: 输入到隐藏层权重，shape (3*hidden_size, input_size)
    # weight_hh_l[k]: 隐藏层到隐藏层权重，shape (3*hidden_size, hidden_size)
    with torch.no_grad():
        for layer in range(numLayers):
            layer_input_size = inputSize if layer == 0 else hiddenSize * num_directions
            gru_layer.weight_ih_l[layer][:, :layer_input_size].copy_(
                weight_ih[layer * 3 * hiddenSize:(layer + 1) * 3 * hiddenSize, :layer_input_size]
            )
            gru_layer.weight_hh_l[layer].copy_(
                weight_hh[layer * 3 * hiddenSize:(layer + 1) * 3 * hiddenSize, :]
            )
            if bias:
                gru_layer.bias_ih_l[layer].copy_(
                    bias_ih[layer * 3 * hiddenSize:(layer + 1) * 3 * hiddenSize]
                )
                gru_layer.bias_hh_l[layer].copy_(
                    bias_hh[layer * 3 * hiddenSize:(layer + 1) * 3 * hiddenSize]
                )

    # 初始化隐藏状态
    if h0 is None:
        h0 = torch.zeros(
            numLayers * num_directions,
            x.shape[1] if not batchFirst else x.shape[0],
            hiddenSize,
            dtype=x.dtype,
            device=x.device
        )

    y, hn = gru_layer(x, h0)
    return y, hn
