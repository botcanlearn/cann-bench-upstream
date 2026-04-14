# LSTM 算子 API 描述

## 1. 算子简介

Long Short-Term Memory 循环神经网络算子，通过输入门、遗忘门、输出门和细胞状态的门控机制实现长距离依赖建模，支持多层堆叠、双向处理、投影降维和可选偏置。

**主要应用场景**：
- 自然语言处理中的序列到序列建模（机器翻译、文本生成）
- 语音识别与合成中的时序特征建模
- 时间序列预测与长距离依赖建模
- 需要投影降维（proj_size）的大规模隐藏状态场景

**算子特征**：
- 难度等级：L4（FusedComposite）
- 多输入（x, weight_ih, weight_hh, 可选 bias_ih, bias_hh, h0, c0）三输出（y, hn, cn）
- 支持多层堆叠、双向处理、batch_first 格式、层间 Dropout、投影降维

## 2. 算子定义

### 数学公式

对于每个时间步 $t$：

$$
i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i) \quad \text{（输入门）}
$$

$$
f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f) \quad \text{（遗忘门）}
$$

$$
g_t = \tanh(W_g x_t + U_g h_{t-1} + b_g) \quad \text{（候选细胞状态）}
$$

$$
o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o) \quad \text{（输出门）}
$$

$$
c_t = f_t \odot c_{t-1} + i_t \odot g_t \quad \text{（细胞状态）}
$$

$$
h_t = o_t \odot \tanh(c_t) \quad \text{（隐藏状态）}
$$

其中：
- $i_t, f_t, o_t$ 分别为输入门、遗忘门、输出门
- $g_t$ 为候选细胞状态
- $c_t$ 为细胞状态，$h_t$ 为隐藏状态
- $\sigma$ 为 sigmoid 函数，$\odot$ 为逐元素乘法

## 3. 接口规范

### 算子原型

```python
ascend_bench.lstm(Tensor x, Tensor weight_ih, Tensor weight_hh, Tensor? bias_ih, Tensor? bias_hh, Tensor? h0, Tensor? c0, int inputSize, int hiddenSize, int numLayers, bool bias, bool batchFirst, float dropout, bool bidirectional, int projSize) -> (Tensor y, Tensor hn, Tensor cn)
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入序列张量，shape 为 (S, B, input_size) 或 (B, S, input_size) |
| weight_ih | Tensor | 必选 | 输入到隐藏层权重，shape 为 (num_layers * 4 * hiddenSize, inputSize) |
| weight_hh | Tensor | 必选 | 隐藏层到隐藏层权重，shape 为 (num_layers * 4 * hiddenSize, hiddenSize) |
| bias_ih | Tensor | None | 输入到隐藏层偏置（可选） |
| bias_hh | Tensor | None | 隐藏层到隐藏层偏置（可选） |
| h0 | Tensor | None | 初始隐藏状态（可选，默认全 0），shape 为 (num_layers * num_directions, B, hiddenSize) |
| c0 | Tensor | None | 初始细胞状态（可选，默认全 0），shape 为 (num_layers * num_directions, B, hiddenSize) |
| inputSize | int | 必选 | 输入特征维度 |
| hiddenSize | int | 必选 | 隐藏状态特征维度 |
| numLayers | int | 1 | 循环层数 |
| bias | bool | true | 是否使用偏置 |
| batchFirst | bool | false | 输入是否为 (batch, seq, feature) 格式 |
| dropout | float | 0.0 | Dropout 概率（层间） |
| bidirectional | bool | false | 是否双向 LSTM |
| projSize | int | 0 | 投影维度（>0 时启用 LSTM with Projection） |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | (S, B, num_directions * hiddenSize) 或 (B, S, num_directions * hiddenSize) | 与输入 x 相同 | 输出序列 |
| hn | (num_layers * num_directions, B, hiddenSize) | 与输入 x 相同 | 最终隐藏状态 |
| cn | (num_layers * num_directions, B, hiddenSize) | 与输入 x 相同 | 最终细胞状态 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float32 | float32 |
| float16 | float16 |
| bfloat16 | bfloat16 |

### 规则与约束

- 所有输入 Tensor（x, weight_ih, weight_hh, bias_ih, bias_hh, h0, c0）的 dtype 必须一致
- `weight_ih` 的 shape 为 (num_layers * 4 * hiddenSize, inputSize)，因 LSTM 有 4 个门（i, f, g, o）
- `weight_hh` 的 shape 为 (num_layers * 4 * hiddenSize, hiddenSize)
- 当 `bias=true` 时，`bias_ih` 和 `bias_hh` 必须提供
- 当 `bidirectional=true` 时，num_directions=2，否则为 1
- `dropout` 仅在 `numLayers > 1` 时生效，作用于层间（非最后一层）
- `projSize > 0` 时启用投影降维，隐藏状态的有效维度变为 projSize
- `batchFirst=true` 时，输入 x 的 shape 为 (B, S, input_size)，输出 y 的 shape 为 (B, S, num_directions * hiddenSize)

## 4. 精度要求

计算结果与 PyTorch Golden 实现逐元素对比，需满足以下误差阈值：

| 数据类型 | 验证方式 | rtol | atol |
|---------|---------|------|------|
| float16 | 相对误差 | 1e-3 | 1e-3 |
| float32 | 相对误差 | 1e-4 | 1e-4 |
| bfloat16 | 相对误差 | 4e-3 | 4e-3 |

**对比公式**：

$$
|output - golden| \leq atol + rtol \times |golden|
$$

## 5. 标准 Golden 代码

```python
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
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import ascend_bench

# 单层单向 LSTM
seq_len, batch, input_size, hidden_size = 20, 8, 128, 256
x = torch.randn(seq_len, batch, input_size, dtype=torch.float32, device="npu")
weight_ih = torch.randn(4 * hidden_size, input_size, dtype=torch.float32, device="npu")
weight_hh = torch.randn(4 * hidden_size, hidden_size, dtype=torch.float32, device="npu")
bias_ih = torch.randn(4 * hidden_size, dtype=torch.float32, device="npu")
bias_hh = torch.randn(4 * hidden_size, dtype=torch.float32, device="npu")
y, hn, cn = ascend_bench.lstm(x, weight_ih, weight_hh, bias_ih, bias_hh, None, None,
                               inputSize=input_size, hiddenSize=hidden_size, numLayers=1,
                               bias=True, batchFirst=False, dropout=0.0,
                               bidirectional=False, projSize=0)

# 双向多层 LSTM with Projection
y2, hn2, cn2 = ascend_bench.lstm(x, weight_ih, weight_hh, bias_ih, bias_hh, None, None,
                                  inputSize=input_size, hiddenSize=hidden_size, numLayers=2,
                                  bias=True, batchFirst=False, dropout=0.1,
                                  bidirectional=True, projSize=64)
```

### 性能基线参考

当前暂无测试用例和性能基线数据。

### 相关算子

- **GRU**：Gated Recurrent Unit 循环神经网络，结构类似但无细胞状态，参数更少
- **MlaProlog**：Multi-Head Latent Attention 前处理，同为 L4 级融合复合算子
- **SparseFlashAttention**：稀疏注意力计算，同为序列处理相关的 L4 级算子
