# DequantSwigluQuant 算子 API 描述

## 1. 算子简介

反量化、SwiGLU和量化的融合。

**主要应用场景**：
- 大语言模型推理中 FFN 层的量化加速
- 低精度推理流水线中 SwiGLU 激活函数的融合计算
- INT8 量化模型中的反量化-激活-重量化一体化操作

**算子特征**：
- 难度等级：L3（FusedComposite）
- 单输入单输出，融合反量化、SwiGLU 激活和量化三个操作
- 输入最后一维 H 必须为偶数（SwiGLU 将其等分为两半）

## 2. 算子定义

### 数学公式

$$
y = \text{quantize}(\text{SwiGLU}(\text{dequantize}(x)))
$$

具体步骤：

1. **反量化**：将 int8 输入转换为浮点数
2. **SwiGLU 激活**：将最后一维等分为两半，当 activate_left=False 时 $y = \text{SiLU}(x_{left}) \times x_{right}$，当 activate_left=True 时 $y = x_{left} \times \text{SiLU}(x_{right})$
3. **量化**：将结果量化为 INT8

## 3. 接口规范

### 算子原型

```python
cann_bench.dequant_swiglu_quant(Tensor x, bool activate_left, str quant_mode, int dst_type) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入张量，任意维度，最后一维 H 必须为偶数 |
| activate_left | bool | False | 是否激活左侧 |
| quant_mode | str | "static" | 量化模式 |
| dst_type | int | 0 | 目标数据类型 (0:DT_INT8) |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 除最后一维减半外与输入相同 | int8 | 输出张量 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | int8 |
| bfloat16 | int8 |
| int8 | int8 |

### 规则与约束

- 输入张量的最后一维 H 必须为偶数，SwiGLU 将其等分为两半分别处理
- activate_left 控制 SwiGLU 的激活方向：False 时对左半部分应用 SiLU，True 时对右半部分应用 SiLU
- quant_mode 指定量化模式，默认 "static"
- dst_type 取值：0 表示 DT_INT8
- 当输入为 int8/int32 类型时，先以 scale=0.1 进行反量化转浮点

## 4. 精度要求

计算结果与 PyTorch Golden 实现逐元素对比：

| 数据类型 | 验证方式 | 阈值 |
|---------|---------|------|
| int8（dst_type=0） | 允许量化边界 off-by-1，最大绝对偏差 ≤ 1；off-by-1 元素占比 | < 1e-4 |

**说明**：y = round(SwiGLU(dequant(x)) × scale) 的 int8 输出允许因 float32 累加顺序差异在 round 时舍到相邻整数；出现 |Δ|≥2 的元素直接判负。

## 5. 标准 Golden 代码

```python
import torch

"""
DequantSwigluQuant算子Torch Golden参考实现

反量化、SwiGLU和量化的融合
公式: y = quantize(SwiGLU(dequantize(x)))
"""
def dequant_swiglu_quant(
    x: torch.Tensor, activate_left: bool = False, quant_mode: str = 'static', dst_type: int = 0
) -> torch.Tensor:
    """
    反量化、SwiGLU和量化的融合
    
    公式: y = quantize(SwiGLU(dequantize(x)))
    
    Args:
        x: 输入张量
        activate_left: 是否激活左侧
        quant_mode: 量化模式'
        dst_type: 目标数据类型 (0:DT_INT8)
    
    Returns:
        输出张量
    """

    def swiglu(x, activate_left=False):
        if activate_left:
            x_left = x[..., :x.shape[-1]//2]
            x_right = x[..., x.shape[-1]//2:]
            return x_left * torch.nn.functional.silu(x_right)
        else:
            x_left = x[..., :x.shape[-1]//2]
            x_right = x[..., x.shape[-1]//2:]
            return torch.nn.functional.silu(x_left) * x_right
    
    if x.dtype in [torch.int8, torch.int32]:
        scale = 0.1
        x_float = x.float() * scale
    else:
        x_float = x
    
    result = swiglu(x_float, activate_left)
    
    # INT8 量化
    scale = (127.0 / result.abs().max()).to(torch.float32)
    y = torch.clamp((result.float() * scale.item()).round(), -128, 127).to(torch.int8)

    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.randn(2, 4096, dtype=torch.float16, device="npu")
y = cann_bench.dequant_swiglu_quant(x, activate_left=False, quant_mode='static', dst_type=0)  # INT8 量化

x_int8 = torch.randint(-128, 127, (2, 4096), dtype=torch.int8, device="npu")
y = cann_bench.dequant_swiglu_quant(x_int8, activate_left=True, quant_mode='static', dst_type=0)  # 反量化后 SwiGLU 再量化
```

### 性能基线参考

基于 cases.yaml 中 20 个测试用例，所有用例的 baseline_perf_us 均为 None，性能基线数据尚未测量。

### 相关算子

- **AddRmsNormDynamicQuant**：同为融合算子，包含 Add、RMSNorm 和动态量化
- **SwiGLU**：L1 级别的 SwiGLU 激活函数算子
- **MoeFinalizeRouting**：L3 级别的融合复合算子
