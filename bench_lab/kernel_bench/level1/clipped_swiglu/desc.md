# ClippedSwiglu 算子 API 描述

## 1. 算子简介

带截断的 Swish 门控线性单元（Clipped SwiGLU）激活函数。相较于标准 SwiGLU，新增了门限截断（limit）、缩放因子（alpha）、偏差（bias）、分组索引（group_index）以及切分方式（interleaved）等参数，用于支持 GPT-OSS 模型使用的变体 SwiGLU 以及 MoE 模型的分组场景。

**主要应用场景**：
- GPT-OSS 等大模型中的变体 SwiGLU 激活
- MoE（Mixture of Experts）模型的分组门控激活
- 需要截断控制的激活函数场景

**算子特征**：
- 难度等级：L1（Elementwise）
- 1 个必选输入 + 1 个可选输入，1 个输出，5 个属性参数
- 支持 ND 格式输入
- 支持奇偶切分（interleaved）和前后切分两种模式
- 支持 FLOAT16 和 BFLOAT16 数据类型

## 2. 算子定义

### 数学公式

对给定的输入张量 x（维度为 [d0, d1, ..., dn]），以 dim 轴为切分轴：

1. **合轴**：将 x 基于 dim 合并为 [pre, cut, after]，然后合并 cut 与 after → [pre, cut*after]

2. **分组过滤**（可选）：根据 group_index 过滤 pre 轴
$$
sum = \text{Sum}(group\_index), \quad x = x[:sum, :]
$$

3. **切分**：
  - interleaved=true（奇偶切分）：$A = x[:, ::2], \ B = x[:, 1::2]$
  - interleaved=false（前后切分）：$A = x[:, :h], \ B = x[:, h:]$，其中 $h = cut*after / 2$

4. **变体 SwiGLU 计算**：
$$
A = A.clamp(max=limit)
$$
$$
B = B.clamp(-limit, limit)
$$
$$
y = A \cdot sigmoid(alpha \cdot A) \cdot (B + bias)
$$

5. **重塑**：y 的维度与原始 x 一致，dim 轴大小为 x 的一半

## 3. 接口规范

### 算子原型

```python
cann_bench.clipped_swiglu(Tensor x, Tensor group_index=None, int64 dim=-1, float alpha=1.702, float limit=7.0, float bias=1.0, bool interleaved=true) -> Tensor y
```

### 输入参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
|--------|----------|------|---------|---------|
| x | 输入 | 输入张量，dim 对应维度必须是偶数 | FLOAT16 / BFLOAT16 | ND |
| group_index | 可选输入 | MoE 分组索引，1 维且元素个数 ≤ 1024 | INT64 | ND |
| dim | 属性 | 合轴及切分的维度序号，取值范围 [-x.dim(), x.dim()-1]，默认 -1 | INT64 | - |
| alpha | 属性 | 变体 SwiGLU 参数，默认 1.702 | FLOAT | - |
| limit | 属性 | 门限值，必须 > 0，默认 7.0 | FLOAT | - |
| bias | 属性 | 偏差参数，默认 1.0 | FLOAT | - |
| interleaved | 属性 | true=奇偶切分，false=前后切分，默认 true | BOOL | - |

### 输出

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
|--------|----------|------|---------|---------|
| y | 输出 | 输出张量，dim 对应维度为 x 的一半 | 与 x 一致 | ND |

### 规则与约束

- x 在 dim 对应维度上必须是偶数
- group_index 为 1 维，元素个数 ≤ 1024
- limit 必须 > 0
- 支持 FLOAT16、BFLOAT16、FLOAT32 的 x，但仅 FLOAT16 和 BFLOAT16 场景需要性能验证

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| x 维度数 | 1 ~ 8 | ND 格式 |
| dim | [-x.dim(), x.dim()-1] | 默认 -1 |
| group_index 长度 | ≤ 1024 | 可选输入 |
| alpha | > 0 | 默认 1.702 |
| limit | > 0 | 默认 7.0 |
| interleaved | true / false | 默认 true |

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
from typing import Optional


def clipped_swiglu(x: torch.Tensor, group_index: Optional[torch.Tensor] = None,
                    dim: int = -1, alpha: float = 1.702,
                    limit: float = 7.0, bias: float = 1.0,
                    interleaved: bool = True):
    """
    带截断的 Swish 门控线性单元激活函数。

    Args:
        x: 输入张量，数据类型为 float16 或 bfloat16
        group_index: 可选，MoE 分组索引，1 维 INT64 张量，None 表示不使用分组
        dim: 合轴及切分的维度序号，默认 -1
        alpha: SwiGLU 参数，默认 1.702
        limit: 门限值，默认 7.0
        bias: 偏差参数，默认 1.0
        interleaved: true=奇偶切分，false=前后切分，默认 true

    Returns:
        y: 输出张量，dim 对应维度为 x 的一半，数据类型与 x 一致
    """
    dim = dim if dim >= 0 else dim + x.dim()
    dtype = x.dtype
    if x.dtype in [torch.bfloat16, torch.float16]:
        x = x.to(torch.float32)
    shape = list(x.shape)
    if x.ndim > 1:
        if dim != 0:
            dim1 = int(torch.prod(torch.tensor(shape[:dim])).item())
            x = x.reshape(dim1, int(torch.prod(torch.tensor(shape[dim:])).item())).clone()
        else:
            x = x.reshape(1, int(torch.prod(torch.tensor(shape[dim:])).item())).clone()
    else:
        x = x.reshape(1, shape[0]).clone()
    group = x.shape[0]
    if group_index is not None:
        group = min(int(torch.sum(group_index).item()), x.shape[0])
    x_tensor = x[:group]
    remain_tensor = torch.zeros_like(x[group:, :x.shape[1] // 2])
    if interleaved:
        x_glu = x_tensor[..., ::2]
        x_linear = x_tensor[..., 1::2]
    else:
        out = torch.chunk(x_tensor, 2, dim=-1)
        x_glu = out[0]
        x_linear = out[1]
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    sigmoid_part = torch.sigmoid(alpha * x_glu)
    result = x_glu * sigmoid_part * (x_linear + bias)
    result = torch.cat((result, remain_tensor), dim=0)
    res_shape = list(shape)
    res_shape[dim] = res_shape[dim] // 2
    if result.numel() != 0:
        result = result.reshape(res_shape)
    return result.to(dtype)
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.randn(20, 20, 4, 4, 48, 120, dtype=torch.float16, device="npu")
group_index = torch.randint(1, 122, (53,), dtype=torch.int64, device="npu")
out = cann_bench.clipped_swiglu(
    x,
    group_index=group_index,
    dim=-2,
    alpha=1.0,
    limit=3.0,
    bias=1.0,
    interleaved=False,
)
# out shape: [20, 20, 4, 4, 48, 60]
```
