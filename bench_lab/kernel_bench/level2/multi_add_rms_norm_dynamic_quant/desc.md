# MultiAddRmsNormDynamicQuant 算子 API 描述

## 1. 算子简介

融合 Add + RmsNorm + DynamicQuant 的算子。在 DeepSeek 推理 MoE 架构中，通信后需要将多个路由专家/共享/机间 hidden state 累加，然后执行 AddRmsNorm + DynamicQuant。本算子将 RmsNorm 前的 n 个 Add 算子和 RmsNorm 归一化输出给到的 DynamicQuant 算子融合起来，减少搬入搬出操作（n 支持 1 到 5）。

**主要应用场景**：
- DeepSeek MoE 推理场景中通信后的 hidden state 融合归一化与量化
- 大模型推理中需要多路 Add + RmsNorm + DynamicQuant 的场景

**算子特征**：
- 难度等级：L2（Normalization + Quantization 融合）
- 3~9 输入（x1 列表 1~5 个 + x2 + gamma + 可选 smooth_scale1 + 可选 smooth_scale2），6 输出
- 支持 ND 格式输入，支持 2D 及以上维度
- 可选属性：epsilon
- 3 种 tiling 策略（normal / single_row / cut_d）

## 2. 算子定义

### 数学公式

**Step 1: 多路加法**

$$
x_1 = x_{1a} + x_{1b_{opt}} + x_{1c_{opt}} + x_{1d_{opt}} + x_{1e_{opt}}
$$

**Step 2: 残差连接**

$$
x = x_1 + x_2
$$

**Step 3: RmsNorm 归一化**

$$
y = \operatorname{RmsNorm}(x) = \frac{x}{\operatorname{Rms}(x)} \cdot \gamma, \quad \operatorname{Rms}(x) = \sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2 + \epsilon}
$$

**Step 4: 动态量化**

- 若 smoothScale1 和 smoothScale2 均不输入：

$$
scale_1 = \frac{\text{row\_max}(|y|)}{127}, \quad y_1 = \text{round}\left(\frac{y}{scale_1}\right)
$$

- 若仅输入 smoothScale1：

$$
input_1 = y \cdot smoothScale_1, \quad scale_1 = \frac{\text{row\_max}(|input_1|)}{127}, \quad y_1 = \text{round}\left(\frac{input_1}{scale_1}\right)
$$

- 若 smoothScale1 和 smoothScale2 均输入：

$$
input_1 = y \cdot smoothScale_1, \quad input_2 = y \cdot smoothScale_2
$$

$$
scale_i = \frac{\text{row\_max}(|input_i|)}{127}, \quad y_i = \text{round}\left(\frac{input_i}{scale_i}\right), \quad i = 1, 2
$$

## 3. 接口规范

### 算子原型

```python
cann_bench.multi_add_rms_norm_dynamic_quant(Tensor t0, Tensor t1, Tensor t2, Tensor t3=None, Tensor t4=None, Tensor t5=None, Tensor t6=None, Tensor t7=None, Tensor t8=None, float epsilon=1e-6, int64 x1_count=1) -> (Tensor y1, Tensor y2, Tensor x, Tensor y, Tensor scale1, Tensor scale2)
```

### 输入参数说明

输入 tensor 按以下顺序排列（由 x1_count 属性确定 x1 列表长度）：

| 参数 | 描述 | 数据类型 | 数据格式 |
|------|------|------|---------|---------|
| x1 列表 | 标准化过程中的源数据张量列表（1~5 个），对应公式中 x1a~x1e | FLOAT16, BFLOAT16 | ND |
| x2 | 标准化过程中的源数据张量，shape 同 x1 | FLOAT16, BFLOAT16 | ND |
| gamma | 归一化权重张量，shape [D]，D 为 x1 最后一维 | FLOAT16, BFLOAT16 | ND |
| smooth_scale1 | 可选，量化路径 1 的 smooth 系数，shape [D] | FLOAT16, BFLOAT16 | ND |
| smooth_scale2 | 可选，量化路径 2 的 smooth 系数，shape [D] | FLOAT16, BFLOAT16 | ND |

| 属性 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| epsilon | float | 1e-6 | 防止除 0 的小常数 |

### 输出

| 参数名 | 描述 | 数据类型 | 精度比较 |
|--------|------|---------|---------|
| y1 | 第一路量化输出 | INT8 | 是 |
| y2 | 第二路量化输出（smooth2 不存在时无意义） | INT8 | 否 |
| x | sum(x1_list) + x2 的和 | FLOAT16, BFLOAT16 | 是 |
| y | RmsNorm 归一化结果 | FLOAT16, BFLOAT16 | 是 |
| scale1 | 第一路量化 scale，per-row | FLOAT32 | 是 |
| scale2 | 第二路量化 scale（smooth2 不存在时无意义） | FLOAT32 | 否 |

### 数据类型

| x1/x2/gamma/smooth dtype | y1/y2 dtype | x/y dtype | scale1/scale2 dtype |
|--------------------------|-------------|-----------|---------------------|
| float16 | int8 | float16 | float32 |
| bfloat16 | int8 | bfloat16 | float32 |

### 规则与约束

- x1 列表中所有 tensor 的 shape 和 dtype 必须相同，且与 x2 的 shape 一致
- gamma 为 1D tensor，长度等于 x1/x2 最后一维大小 D
- smooth_scale1、smooth_scale2 为 1D tensor，长度等于 D，可选
- 若 smooth_scale2 存在，则 smooth_scale1 必须存在
- 所有输入 tensor dtype 必须一致（同为 float16 或同为 bfloat16）
- 支持 Atlas A2/A3 训练/推理系列产品（ascend910b, ascend910_93）

### 支持范围

| 参数 | 范围 | 备注 |
|------|------|------|
| x1_count | 1 ~ 5 | x1 列表长度 |
| 输入维度 | 2D 及以上 | ND 格式 |
| D（最后一维） | 任意 | gamma/smooth 长度 |
| epsilon | 任意正浮点数 | 默认 1e-6 |

## 4. 精度要求

采用[生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)进行验证。

**误差指标**：

1. 平均相对误差（MERE）：

   $$
   \text{MERE} = \text{avg}\left(\frac{|actual - golden|}{|golden| + 10^{-7}}\right)
   $$

2. 最大相对误差（MARE）：

   $$
   \text{MARE} = \max\left(\frac{|actual - golden|}{|golden| + 10^{-7}}\right)
   $$

**通过标准**：

| 数据类型 | FLOAT16 | BFLOAT16 | FLOAT32 |
|----------|---------|----------|---------|
| **通过阈值** | 2^-10 | 2^-7 | 2^-13 |

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过。

注意：y1/y2 为 INT8 量化输出，精度标准参考对应输入 dtype（float16/bfloat16）。y2 和 scale2 在 smooth2 不存在时不参与精度比较。

## 5. 标准 Golden 代码

```python
import torch

def multi_add_rms_norm_dynamic_quant(
    t0: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor,
    t3: torch.Tensor = None, t4: torch.Tensor = None,
    t5: torch.Tensor = None, t6: torch.Tensor = None,
    t7: torch.Tensor = None, t8: torch.Tensor = None,
    epsilon: float = 1e-6, x1_count: int = 1
):
    all_tensors = [t for t in [t0, t1, t2, t3, t4, t5, t6, t7, t8] if t is not None]
    x1_list = all_tensors[:x1_count]
    remaining = all_tensors[x1_count:]
    x2, gamma = remaining[0], remaining[1]
    smooth1 = remaining[2] if len(remaining) > 2 else None
    smooth2 = remaining[3] if len(remaining) > 3 else None

    ori_dtype = x2.dtype
    x1_sum = torch.zeros_like(x2, dtype=torch.float32)
    for x1_t in x1_list:
        x1_sum = x1_sum + x1_t.float()
    x_sum = x1_sum + x2.float()

    rstd = torch.rsqrt(x_sum.pow(2).mean(dim=-1, keepdim=True) + epsilon)
    y_fp32 = x_sum * rstd * gamma.float()

    input1 = y_fp32 * smooth1.float() if smooth1 is not None else y_fp32
    x_max1 = torch.max(torch.abs(input1), dim=-1, keepdim=True)[0]
    y1 = torch.round(input1 * (127.0 / x_max1)).to(torch.int8)
    scale1 = (x_max1 / 127.0).squeeze(-1)

    if smooth2 is not None:
        input2 = y_fp32 * smooth2.float()
        x_max2 = torch.max(torch.abs(input2), dim=-1, keepdim=True)[0]
        y2 = torch.round(input2 * (127.0 / x_max2)).to(torch.int8)
        scale2 = (x_max2 / 127.0).squeeze(-1)
    else:
        y2 = torch.zeros_like(y1)
        scale2 = torch.zeros_like(scale1)

    return y1, y2, x_sum.to(ori_dtype), y_fp32.to(ori_dtype), scale1.float(), scale2.float()
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

# x1_count=1, with smooth1 and smooth2
x1a = torch.randn(2048, 7680, dtype=torch.bfloat16, device="npu")
x2 = torch.randn(2048, 7680, dtype=torch.bfloat16, device="npu")
gamma = torch.randn(7680, dtype=torch.bfloat16, device="npu")
smooth1 = torch.randn(7680, dtype=torch.bfloat16, device="npu")
smooth2 = torch.randn(7680, dtype=torch.bfloat16, device="npu")

y1, y2, x, y, scale1, scale2 = cann_bench.multi_add_rms_norm_dynamic_quant(
    x1a, x2, gamma, smooth1, smooth2, epsilon=1e-6, x1_count=1
)
```
