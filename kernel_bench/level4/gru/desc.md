# GRU 算子 API 描述

## 1. 算子简介

Gated Recurrent Unit 循环神经网络算子，实现带门控机制的循环单元，通过更新门和重置门控制信息流动，支持多层堆叠、双向处理和可选偏置。

**主要应用场景**：
- 自然语言处理中的序列建模（机器翻译、文本分类）
- 语音识别中的时序特征提取
- 时间序列预测与异常检测
- 作为 LSTM 的轻量替代方案（参数更少，无细胞状态）

**算子特征**：
- 难度等级：L4（FusedComposite）
- 多输入（x, weight_ih, weight_hh, 可选 bias_ih, bias_hh, h0）双输出（y, hn）
- 支持多层堆叠、双向处理、batch_first 格式、层间 Dropout

## 2. 算子定义

### 数学公式

对于每个时间步 $t$：

$$
z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)
$$

$$
r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)
$$

$$
n_t = \tanh(W_n x_t + r_t \odot (U_n h_{t-1} + b_n))
$$

$$
h_t = (1 - z_t) \odot n_t + z_t \odot h_{t-1}
$$

其中：
- $z_t$ 为更新门，控制前一时刻隐藏状态的保留比例
- $r_t$ 为重置门，控制前一时刻隐藏状态对候选状态的影响
- $n_t$ 为候选隐藏状态
- $h_t$ 为当前时刻的隐藏状态
- $\sigma$ 为 sigmoid 函数，$\odot$ 为逐元素乘法

## 3. 接口规范

### 算子原型

```python
ascend_bench.gru(Tensor x, Tensor weight_ih, Tensor weight_hh, Tensor? bias_ih, Tensor? bias_hh, Tensor? h0, int inputSize, int hiddenSize, int numLayers, bool bias, bool batchFirst, float dropout, bool bidirectional) -> (Tensor y, Tensor hn)
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入序列张量，shape 为 (S, B, input_size) 或 (B, S, input_size) |
| weight_ih | Tensor | 必选 | 输入到隐藏层权重，shape 为 (num_layers * 3 * hiddenSize, inputSize) |
| weight_hh | Tensor | 必选 | 隐藏层到隐藏层权重，shape 为 (num_layers * 3 * hiddenSize, hiddenSize) |
| bias_ih | Tensor | None | 输入到隐藏层偏置（可选） |
| bias_hh | Tensor | None | 隐藏层到隐藏层偏置（可选） |
| h0 | Tensor | None | 初始隐藏状态（可选，默认全 0），shape 为 (num_layers * num_directions, B, hiddenSize) |
| inputSize | int | 必选 | 输入特征维度 |
| hiddenSize | int | 必选 | 隐藏状态特征维度 |
| numLayers | int | 1 | 循环层数 |
| bias | bool | true | 是否使用偏置 |
| batchFirst | bool | false | 输入是否为 (batch, seq, feature) 格式 |
| dropout | float | 0.0 | Dropout 概率 |
| bidirectional | bool | false | 是否双向 GRU |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | (S, B, num_directions * hiddenSize) 或 (B, S, num_directions * hiddenSize) | 与输入 x 相同 | 输出序列 |
| hn | (num_layers * num_directions, B, hiddenSize) | 与输入 x 相同 | 最终隐藏状态 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float32 | float32 |
| float16 | float16 |
| bfloat16 | bfloat16 |

### 规则与约束

- 所有输入 Tensor（x, weight_ih, weight_hh, bias_ih, bias_hh, h0）的 dtype 必须一致
- `weight_ih` 的 shape 为 (num_layers * 3 * hiddenSize, inputSize)，因 GRU 有 3 个门（z, r, n）
- `weight_hh` 的 shape 为 (num_layers * 3 * hiddenSize, hiddenSize)
- 当 `bias=true` 时，`bias_ih` 和 `bias_hh` 必须提供
- 当 `bidirectional=true` 时，num_directions=2，否则为 1
- `dropout` 仅在 `numLayers > 1` 时生效，作用于层间（非最后一层）
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
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import ascend_bench

# 单层单向 GRU
seq_len, batch, input_size, hidden_size = 20, 8, 128, 256
x = torch.randn(seq_len, batch, input_size, dtype=torch.float32, device="npu")
weight_ih = torch.randn(3 * hidden_size, input_size, dtype=torch.float32, device="npu")
weight_hh = torch.randn(3 * hidden_size, hidden_size, dtype=torch.float32, device="npu")
bias_ih = torch.randn(3 * hidden_size, dtype=torch.float32, device="npu")
bias_hh = torch.randn(3 * hidden_size, dtype=torch.float32, device="npu")
y, hn = ascend_bench.gru(x, weight_ih, weight_hh, bias_ih, bias_hh, None,
                          inputSize=input_size, hiddenSize=hidden_size, numLayers=1,
                          bias=True, batchFirst=False, dropout=0.0, bidirectional=False)

# batch_first 模式
x_bf = torch.randn(batch, seq_len, input_size, dtype=torch.float32, device="npu")
y_bf, hn_bf = ascend_bench.gru(x_bf, weight_ih, weight_hh, bias_ih, bias_hh, None,
                                inputSize=input_size, hiddenSize=hidden_size, numLayers=1,
                                bias=True, batchFirst=True, dropout=0.0, bidirectional=False)
```

### 性能基线参考

当前暂无测试用例和性能基线数据。

### 相关算子

- **LSTM**：Long Short-Term Memory 循环神经网络，结构类似但增加了细胞状态和输出门
- **MlaProlog**：Multi-Head Latent Attention 前处理，同为 L4 级融合复合算子
- **SparseFlashAttention**：稀疏注意力计算，同为序列处理相关的 L4 级算子
