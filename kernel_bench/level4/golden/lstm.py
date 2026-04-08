import torch

"""
LSTM 算子 Torch Golden 参考实现

Long Short-Term Memory 循环神经网络
公式:
    i_t = σ(W_i @ x_t + U_i @ h_{t-1} + b_i)  # 输入门
    f_t = σ(W_f @ x_t + U_f @ h_{t-1} + b_f)  # 遗忘门
    g_t = tanh(W_g @ x_t + U_g @ h_{t-1} + b_g)  # 候选细胞状态
    o_t = σ(W_o @ x_t + U_o @ h_{t-1} + b_o)  # 输出门
    c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t  # 细胞状态
    h_t = o_t ⊙ tanh(c_t)  # 隐藏状态
"""
def lstm(
    x: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor | None = None,
    bias_hh: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    c0: torch.Tensor | None = None,
    inputSize: int = 0,
    hiddenSize: int = 0,
    numLayers: int = 1,
    bias: bool = True,
    batchFirst: bool = False,
    dropout: float = 0.0,
    bidirectional: bool = False,
    projSize: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Long Short-Term Memory 循环神经网络

    Args:
        x: 输入序列张量 (S, B, input_size) 或 (B, S, input_size) if batch_first
        weight_ih: 输入到隐藏层权重 (num_layers * 4 * hiddenSize, inputSize)
        weight_hh: 隐藏层到隐藏层权重 (num_layers * 4 * hiddenSize, hiddenSize)
        bias_ih: 输入到隐藏层偏置 (可选)
        bias_hh: 隐藏层到隐藏层偏置 (可选)
        h0: 初始隐藏状态 (可选)
        c0: 初始细胞状态 (可选)
        inputSize: 输入特征维度
        hiddenSize: 隐藏状态特征维度
        numLayers: 循环层数
        bias: 是否使用偏置
        batchFirst: 输入是否为 (batch, seq, feature) 格式
        dropout: Dropout 概率
        bidirectional: 是否双向 LSTM
        projSize: 投影维度（>0 时启用 LSTM with Projection）

    Returns:
        y: 输出序列
        hn: 最终隐藏状态
        cn: 最终细胞状态
    """
    num_directions = 2 if bidirectional else 1
    effective_hidden_size = projSize if projSize > 0 else hiddenSize

    # 使用 torch.nn.LSTM 实现
    lstm_layer = torch.nn.LSTM(
        input_size=inputSize,
        hidden_size=hiddenSize,
        num_layers=numLayers,
        bias=bias,
        batch_first=batchFirst,
        dropout=dropout if numLayers > 1 else 0.0,
        bidirectional=bidirectional,
        proj_size=projSize if projSize > 0 else 0
    )

    # 手动设置权重以匹配传入的 weight
    # PyTorch LSTM 权重格式：
    # weight_ih_l[k]: 输入到隐藏层权重，shape (4*hidden_size, input_size)
    # weight_hh_l[k]: 隐藏层到隐藏层权重，shape (4*hidden_size, hidden_size)
    with torch.no_grad():
        for layer in range(numLayers):
            layer_input_size = inputSize if layer == 0 else effective_hidden_size * num_directions
            lstm_layer.weight_ih_l[layer][:, :layer_input_size].copy_(
                weight_ih[layer * 4 * hiddenSize:(layer + 1) * 4 * hiddenSize, :layer_input_size]
            )
            lstm_layer.weight_hh_l[layer].copy_(
                weight_hh[layer * 4 * hiddenSize:(layer + 1) * 4 * hiddenSize, :]
            )
            if bias:
                lstm_layer.bias_ih_l[layer].copy_(
                    bias_ih[layer * 4 * hiddenSize:(layer + 1) * 4 * hiddenSize]
                )
                lstm_layer.bias_hh_l[layer].copy_(
                    bias_hh[layer * 4 * hiddenSize:(layer + 1) * 4 * hiddenSize]
                )

    # 初始化隐藏状态和细胞状态
    batch_size = x.shape[1] if not batchFirst else x.shape[0]
    if h0 is None:
        h0 = torch.zeros(
            numLayers * num_directions,
            batch_size,
            effective_hidden_size,
            dtype=x.dtype,
            device=x.device
        )
    if c0 is None:
        c0 = torch.zeros(
            numLayers * num_directions,
            batch_size,
            hiddenSize,
            dtype=x.dtype,
            device=x.device
        )

    y, (hn, cn) = lstm_layer(x, (h0, c0))
    return y, hn, cn
