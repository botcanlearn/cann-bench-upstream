# SwiGlu 算子 API 描述

## 1. 算子简介

SwiGlu 是采用 Swish 作为激活函数的 GLU（Gated Linear Unit）变体，输入在最后一维拆分成 x0 和 x1 两部分，x0 经 Swish 激活后与 x1 做门控乘法。

**主要应用场景**：
- LLaMA、PaLM 等大语言模型的前馈网络
- Transformer FFN 层中替代传统 ReLU/GELU 的激活方案

**算子特征**：
- 难度等级：L1（Elementwise）
- 单输入单输出，输入在 -1 维拆分为两部分，输出 shape 的最后一维为输入的一半

## 2. 算子定义

### 数学公式

输入 x 沿最后一维拆分为 x0、x1 两等份：

$$
x0, x1 = \text{chunk}(x, 2, \text{dim}=-1)
$$

$$
\text{Swish}(x0) = x0 \cdot \sigma(\beta \cdot x0)
$$

$$
y = \text{Swish}(x0) \cdot x1
$$

其中 $\sigma$ 为 Sigmoid 函数，$\beta$ 为 `scalarValue` 参数。

## 3. 接口规范

### 算子原型

```python
cann_bench.swi_glu(Tensor x, float scalarValue) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入张量，会在 -1 维拆分成 x0 和 x1 |
| scalarValue | float | 必选 | Swish 激活函数的 beta 参数 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 输入 shape 的最后一维除以 2 | 与输入 x 相同 | SwiGlu 门控激活结果 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| bfloat16 | bfloat16 |
| float32 | float32 |

### 规则与约束

- 输出 shape 的最后一维为输入最后一维的一半
- 输出 dtype 与输入 dtype 一致
- 若输入最后一维为奇数，则仅取前偶数个元素进行拆分

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `x` 维度数 (ndim) | 1 ~ 8 | cases.csv 实测 2 ~ 5 |
| `x` 最后一维 `D` | 2 ~ 131072 | cases.csv 实测 2 ~ 65521；输出最后一维为 `D/2`；奇数时仅前 `⌊D/2⌋×2` 个元素参与拆分 |
| `x` 其余维大小 | 1 ~ 2M | cases.csv 实测 2 ~ 1000003 |
| `x` 总元素数 | 1 ~ 256M | cases.csv 实测 ~1M ~ 134M (8192×16384) |
| `scalarValue`（Swish β） | 0.0 ~ 4.0 | cases.csv 实测 0.0 ~ 2.0；允许 nan/特殊值 |

约束：输出最后一维等于 `⌊D/2⌋`；其余维与输入一致。

## 4. 精度要求

采用[生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)进行验证。

**误差指标**：

1. 平均相对误差（MERE）：采样点中相对误差平均值

   $$
   \text{MERE} = \text{avg}(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

2. 最大相对误差（MARE）：采样点中相对误差最大值

   $$
   \text{MARE} = \max(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

**通过标准**：

| 数据类型 | FLOAT16 | BFLOAT16 | FLOAT32 | HiFLOAT32 | FLOAT8 E4M3 | FLOAT8 E5M2 |
|----------|---------|----------|---------|-----------|-------------|-------------|
| **通过阈值(Threshold)** | 2^-10 | 2^-7 | 2^-13 | 2^-11 | 2^-3 | 2^-2 |

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过。


## 5. 标准 Golden 代码

```python
import torch

def swi_glu(
    x: torch.Tensor, scalarValue: float
) -> torch.Tensor:
    """
    采用Swish作为激活函数的GLU变体，输入在第-1维拆分成x0和x1两部分

    公式: y = swish(x0) * x1 = x0 * sigmoid(beta * x0) * x1

    Args:
        x: 输入张量，会在-1维拆分成x0和x1
        scalarValue: Swish激活函数的beta参数

    Returns:
        输出张量，形状为输入shape除以2
    """

    # 在最后一维拆分为两部分
    last_dim_size = x.shape[-1]

    # FP16/BF16需要升高精度到FP32计算
    out_dtype = x.dtype
    x = x.to(torch.float)

    # 对于奇数维度，只取前偶数个元素进行拆分，确保两部分大小一致
    if last_dim_size % 2 != 0:
        # 取前 floor(n/2)*2 个元素
        usable_size = (last_dim_size // 2) * 2
        x = x[..., :usable_size]

    x0, x1 = x.chunk(2, dim=-1)
    swish = x0 * torch.sigmoid(scalarValue * x0)
    y = swish * x1
    return y.to(out_dtype)
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.randn(1024, 1024, dtype=torch.float32, device="npu")
y = cann_bench.swi_glu(x, scalarValue=1.0)
```
