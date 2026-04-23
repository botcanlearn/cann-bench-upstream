# AddRmsNormDynamicQuant 算子 API 描述

## 1. 算子简介

Add、RMSNorm 和动态量化的融合。

**主要应用场景**：
- 大语言模型推理中残差连接 + 归一化 + 量化的融合加速
- Transformer 模型中 RMSNorm 前的残差加法与后处理量化一体化
- INT8/INT4 低精度推理的动态量化预处理

**算子特征**：
- 难度等级：L3（FusedComposite）
- 多输入多输出，融合 Add、RMSNorm 和动态量化三个操作
- 输入 x1、x2 为 ND 格式张量，gamma 为缩放参数

## 2. 算子定义

### 数学公式

$$
y, xOut, scaleOut = \text{quantize}(\text{rmsnorm}(x_1 + x_2) \times \gamma)
$$

具体步骤：

1. **Add 操作**：$xOut = x_1 + x_2$
2. **RMSNorm**：$\text{normalized} = \frac{xOut}{\sqrt{\text{mean}(xOut^2) + \epsilon}} \times \gamma$
3. **动态量化**：根据 dst_type 将归一化结果量化为 INT8 或 INT4，同时输出量化 scale

## 3. 接口规范

### 算子原型

```python
cann_bench.add_rms_norm_dynamic_quant(Tensor x1, Tensor x2, Tensor gamma, float epsilon, int dst_type) -> (Tensor y, Tensor xOut, Tensor scaleOut)
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x1 | Tensor | 必选 | 第 1 个输入张量 |
| x2 | Tensor | 必选 | 第 2 个输入张量 |
| gamma | Tensor | 必选 | 缩放参数 |
| epsilon | float | 1e-6 | epsilon 值 |
| dst_type | int | 0 | 目标数据类型 (0:DT_INT8, 1:DT_INT4) |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 与输入 x1 相同 | int8 / int4 | 量化后的输出张量 |
| xOut | 与输入 x1 相同 | float16 / bfloat16 | Add 结果，x1 + x2 |
| scaleOut | 标量 | float32 | 量化使用的 scale 值 |

### 数据类型

| 输入 (x1, x2, gamma) dtype | 输出 y dtype | 输出 xOut dtype | 输出 scaleOut dtype |
|---------------------------|-------------|----------------|-------------------|
| float16 | int8 / int4 | float16 | float32 |
| bfloat16 | int8 / int4 | bfloat16 | float32 |

**注意**：INT4 量化（dst_type=1）的输出值范围为 [-8, 7]，由于 PyTorch 不支持 int4 dtype，实际存储为 int8 dtype。

### 规则与约束

- x1 和 x2 的 shape 和 dtype 必须一致
- gamma 的 dtype 须与 x1、x2 一致
- x1 为 ND 格式
- dst_type 取值：0 表示 DT_INT8，1 表示 DT_INT4
- epsilon 用于 RMSNorm 的数值稳定性，默认 1e-6
- scaleOut 为 float32 类型标量

## 4. 精度要求

计算结果与 PyTorch Golden 实现逐元素对比：

| 数据类型 | 验证方式 | 阈值 |
|---------|---------|------|
| float16（xOut） | 相对误差：`\|output-golden\| ≤ atol + rtol×\|golden\|` | rtol=1e-3, atol=1e-3 |
| bfloat16（xOut） | 相对误差：`\|output-golden\| ≤ atol + rtol×\|golden\|` | rtol=4e-3, atol=4e-3 |
| float32（scaleOut） | 相对误差 | rtol=1e-3, atol=1e-5 |
| int8（y, dst_type=0） | 允许量化边界 off-by-1，最大绝对偏差 ≤ 1；off-by-1 元素占比 | < 1e-4 |
| int4（y, dst_type=1，packed 为 int8，值域 [-8,7]） | 允许量化边界 off-by-1，最大绝对偏差 ≤ 1；off-by-1 元素占比 | < 1e-4 |

**说明**：量化输出（int8 / int4）允许因 float32 累加顺序差异在 round 时舍到相邻整数；出现 |Δ|≥2 的元素直接判负。`xOut` 与 `scaleOut` 作为非量化通路，按上面的相对误差规则比较。

## 5. 标准 Golden 代码

```python
import torch

"""
AddRmsNormDynamicQuant 算子 Torch Golden 参考实现

Add、RMSNorm 和动态量化的融合
公式：y, xOut, scaleOut = quantize(rmsnorm(x1 + x2) * gamma)
"""
def add_rms_norm_dynamic_quant(
    x1: torch.Tensor,
    x2: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float = 1e-6,
    dst_type: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Add、RMSNorm 和动态量化的融合

    公式：y, xOut, scaleOut = quantize(rmsnorm(x1 + x2) * gamma)

    Args:
        x1: 第 1 个输入张量
        x2: 第 2 个输入张量
        gamma: 缩放参数
        epsilon: epsilon 值
        dst_type: 目标数据类型 (0:DT_INT8, 1:DT_INT4)

    Returns:
        y: 量化后的输出张量
        xOut: Add 结果，x1 + x2
        scaleOut: 量化使用的 scale 值
    """

    # Add 操作
    xOut = x1 + x2

    # RMSNorm
    variance = xOut.pow(2).mean(-1, keepdim=True)
    rms = torch.sqrt(variance + epsilon)
    normalized = xOut / rms
    y_norm = normalized * gamma

    # 动态量化
    # 将 y_norm 转换为 float32 以保证 scale 计算精度和 dtype 正确
    y_norm_f32 = y_norm.float()

    if dst_type == 0:  # INT8
        scale = (127.0 / y_norm_f32.abs().max()).to(torch.float32)
        y = torch.clamp((y_norm_f32 * scale.item()).round(), -128, 127).to(torch.int8)
    else:  # INT4 (存储为 int8，值范围 [-8, 7])
        scale = (7.0 / y_norm_f32.abs().max()).to(torch.float32)
        y = torch.clamp((y_norm_f32 * scale.item()).round(), -8, 7).to(torch.int8)

    return y, xOut, scale
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x1 = torch.randn(2, 4096, dtype=torch.float16, device="npu")
x2 = torch.randn(2, 4096, dtype=torch.float16, device="npu")
gamma = torch.ones(4096, dtype=torch.float16, device="npu")

y, xOut, scaleOut = cann_bench.add_rms_norm_dynamic_quant(x1, x2, gamma, 1e-6, 0)  # INT8 量化
y, xOut, scaleOut = cann_bench.add_rms_norm_dynamic_quant(x1, x2, gamma, 1e-6, 1)  # INT4 量化
```

### 性能基线参考

基于 cases.yaml 中 20 个测试用例，所有用例的 baseline_perf_us 均为 None，性能基线数据尚未测量。

### 相关算子

- **DequantSwigluQuant**：同为融合算子，包含反量化、SwiGLU 激活和量化操作
- **MoeRerouting**：L3 级别的融合复合算子
- **MoeFinalizeRouting**：L3 级别的融合复合算子
