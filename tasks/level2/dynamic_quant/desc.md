# DynamicQuant 算子 API 描述

## 1. 算子简介

对输入张量进行 per-token 对称动态量化。

**主要应用场景**：
- 大语言模型推理加速中的动态量化（W8A8 / W4A8 等方案）
- KV Cache 量化压缩以节省显存
- 模型部署阶段的在线量化处理

**算子特征**：
- 难度等级：L2（FusedComposite）
- 单输入单输出，涉及求最大值、缩放、四舍五入等多步融合计算
- 输入支持 2-8 维张量

## 2. 算子定义

### 数学公式

$$
scaleOut = \frac{\max_{\text{last-dim}}(|x|)}{127}
$$

$$
yOut = \text{round}\left(\frac{x}{scaleOut}\right)
$$

其中：
- $\max_{\text{last-dim}}(|x|)$ 表示沿 last-dim（每个 token）取绝对值最大值
- 量化目标固定为 int8（对应 $dtypeMax = 127$）
- $\text{round}$ 为四舍五入到最近整数（half-to-even）

> **NPU API 约束**：CANN `torch_npu.npu_dynamic_quant` 只支持 per-token 量化（沿 last-dim），不暴露 axis 参数；不支持 float32 输入；不支持 1D 张量。本算子规格与 NPU API 真实能力对齐。

## 3. 接口规范

### 算子原型

```python
cann_bench.dynamic_quant(Tensor x) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入张量，shape ≥ 2 维，dtype ∈ {float16, bfloat16} |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 与输入 x 相同 | int8 | 量化后的张量 |

> **说明**：golden 显式 cast 输出到 int8（与 proto.outputs.dtype 一致），让"数学正确性"和"dtype 合约"两层在 golden 这一层就对齐，精度比对直接走整数 ±1 容差路径。性能层水位由 ref 函数测量，跟 golden 解耦。
> NPU API 实际返回 `(y, scale)` 双输出（`scale` 是 fp32，shape = x.shape[:-1]），但本算子规格只评估 `y`；`scale` 不作为输出参与精度比对。

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | int8 |
| bfloat16 | int8 |

### 规则与约束

- 输入 x 必须为 2 ~ 8 维张量（NPU API 硬性要求 ≥ 2 维）
- 输入 dtype 仅支持 float16 / bfloat16（NPU API 不支持 float32）
- 量化为对称量化（zero_point 恒为 0），scale 基于每 token 绝对值最大值计算
- 输出 shape 与输入 shape 完全一致
- 当某个 token 全部为 0 时 scale = 0，公式产生 NaN；NPU 实现与 golden 一致地输出 NaN（用户应避免此情形）

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

**int8 输出特殊阈值**：

量化算子输出为整数类型，round 操作存在舍入误差，允许 ±1 的绝对误差：

| 输出类型 | 阈值 | 说明 |
|----------|------|------|
| int8 | 1.0 | 允许 \|actual - golden\| ≤ 1 |

**通过条件**：`|actual - golden| ≤ threshold`


## 5. 标准 Golden 代码

```python
import torch

"""
DynamicQuant 算子 Torch Golden 参考实现

per-token 对称动态量化 (沿 last-dim)，对齐 NPU torch_npu.npu_dynamic_quant 默认行为。
公式: scaleOut = row_max(abs(x)) / 127, yOut = round(x / scaleOut)

输出显式 cast 到 int8，与 proto.outputs.dtype 一致；让 golden 在数学上和
dtype 层都明确返回 int8，方便精度比对（性能层水位由 ref 函数测量）。
"""
def dynamic_quant(x: torch.Tensor) -> torch.Tensor:
    """Per-token 对称动态量化 (axis=-1, dtype_max=127→int8)。

    NPU API 只支持沿 last-dim 量化、输入 fp16/bf16、输出 int8 (+scale)。
    此 golden 镜像 NPU API 默认行为，不暴露 axis / dst_type 参数。

    Args:
        x: 输入张量 (fp16/bf16)，shape ≥ 2 维

    Returns:
        量化后的张量 (int8, shape 与 x 一致)
    """
    if x.dtype in (torch.float16, torch.bfloat16):
        x_compute = x.to(torch.float32)
    else:
        x_compute = x

    scale_out = torch.max(torch.abs(x_compute), dim=-1, keepdim=True)[0] / 127.0
    y = torch.round(x_compute / scale_out).to(torch.int8)
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

# 2D per-token quant
x = torch.randn(1024, 1024, dtype=torch.float16, device="npu")
y = cann_bench.dynamic_quant(x)          # y.shape = (1024, 1024), y.dtype = int8

# 4D per-token quant（每行最后维量化）
x = torch.randn(2, 8, 256, 256, dtype=torch.bfloat16, device="npu")
y = cann_bench.dynamic_quant(x)          # y.shape = (2, 8, 256, 256), y.dtype = int8
```
