# Dilation2D 算子 API 描述

## 1. 算子简介

2D形态学膨胀操作，使用最大池化在局部邻域内获取最大值。

**主要应用场景**：
- 图像形态学处理中的膨胀操作
- 目标边缘扩展和连通区域填充
- 医学影像分割中的形态学后处理
- 文字识别中的笔画膨胀增强

**算子特征**：
- 难度等级：L3（Contraction）
- 双输入（图像和结构元素/卷积核）单输出
- 输入 shape 为 [batch, height, width, depth]（NHWC 格式），输出 shape 由 padding 和 stride 决定

## 2. 算子定义

### 数学公式

$$
y[b, y, x, c] = \max_{dy, dx} \left( x[b, y \cdot \text{stride\_h} + \text{rate\_h} \cdot dy, x \cdot \text{stride\_w} + \text{rate\_w} \cdot dx, c] + \text{filter}[dy, dx, c] \right)
$$

对每个输出位置 $(b, y, x, c)$，在以 rates 确定的空洞采样窗口内，计算输入与结构元素逐元素加法，然后取最大值。

## 3. 接口规范

### 算子原型

```python
cann_bench.dilation2_d(Tensor x, Tensor filter, int[] strides, int[] rates, str padding_mode, int[] pads, bool ceil_mode, str data_format) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入图像 |
| filter | Tensor | 必选 | 结构元素/卷积核 |
| strides | int[] | 必选 | 滑动窗口的步长 [1, stride_h, stride_w, 1]，首尾固定为1 |
| rates | int[] | 必选 | 膨胀率 [1, rate_h, rate_w, 1]，首尾固定为1，用于空洞膨胀 |
| padding_mode | str | "SAME" | 填充模式：'SAME' 或 'VALID' |
| pads | int[] | [0, 0, 0, 0] | 填充值 [pad_top, pad_bottom, pad_left, pad_right] |
| ceil_mode | bool | False | 是否向上取整计算输出尺寸 |
| data_format | str | "NHWC" | 数据格式，如 'NHWC' |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 由输入尺寸、strides、rates 和 padding 决定 | 与输入 x 相同 | 膨胀后的图像 |

### 数据类型

| 输入 (x, filter) dtype | 输出 dtype |
|-----------------------|-----------|
| float16 | float16 |

### 规则与约束

- 输入 x 默认为 NHWC 格式，shape 为 [batch, height, width, depth]
- filter 为结构元素，shape 为 [filter_h, filter_w, channels]
- strides 格式为 [1, stride_rows, stride_cols, 1]，首尾维度固定为 1
- rates 格式为 [1, rate_rows, rate_cols, 1]，首尾维度固定为 1，控制空洞膨胀
- padding_mode 支持 'SAME' 和 'VALID' 两种模式
- SAME 模式下自动计算 padding 使输出尺寸为 ceil(input_size / stride)
- VALID 模式下可通过 pads 参数手动指定填充
- ceil_mode 控制输出尺寸计算时是否向上取整

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

"""
Dilation2D 算子 Torch Golden 参考实现

2D形态学膨胀操作，使用最大池化在局部邻域内获取最大值
公式: y[b, y, x, c] = max_{dy,dx} x[b, y + rates[1]*dy, x + rates[2]*dx, c] * filter[dy, dx, c]
"""
def dilation2_d(
    x: torch.Tensor, filter: torch.Tensor, strides: list, rates: list,
    pads: list = [0, 0, 0, 0], padding_mode: str = 'SAME',
    ceil_mode: bool = False, data_format: str = 'NHWC'
) -> torch.Tensor:
    """
    2D形态学膨胀操作，对每个位置在膨胀窗口内取 input + filter 的最大值

    公式: y[b, y, x, c] = max_{dy,dx} (x[b, y*stride_h + rate_h*dy, x*stride_w + rate_w*dx, c] + filter[dy, dx, c])

    Args:
        x: 输入图像，shape 为 [batch, height, width, depth] (NHWC) 或 [batch, depth, height, width] (NCHW)
        filter: 结构元素/卷积核，shape 为 [filter_h, filter_w, depth]
        strides: 步长 [1, stride_h, stride_w, 1]，首尾固定为1
        rates: 膨胀率 [1, rate_h, rate_w, 1]，首尾固定为1
        pads: 填充值 [pad_top, pad_bottom, pad_left, pad_right]
        padding_mode: 填充模式：'SAME' 或 'VALID'
        ceil_mode: 是否向上取整计算输出尺寸
        data_format: 数据格式，'NHWC' 或 'NCHW'

    Returns:
        膨胀后的图像
    """

    if data_format == 'NHWC':
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        filter = filter.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]

    batch, channels, in_h, in_w = x.shape
    filter_h, filter_w = filter.shape[1], filter.shape[2]
    stride_h, stride_w = strides[1], strides[2]
    rate_h, rate_w = rates[1], rates[2]

    effective_filter_h = (filter_h - 1) * rate_h + 1
    effective_filter_w = (filter_w - 1) * rate_w + 1

    if padding_mode == 'SAME':
        out_h = (in_h + stride_h - 1) // stride_h
        out_w = (in_w + stride_w - 1) // stride_w
        if ceil_mode:
            out_h = (in_h + stride_h - 1) // stride_h + (1 if (in_h - 1) % stride_h else 0)
            out_w = (in_w + stride_w - 1) // stride_w + (1 if (in_w - 1) % stride_w else 0)
        pad_h = max((out_h - 1) * stride_h + effective_filter_h - in_h, 0)
        pad_w = max((out_w - 1) * stride_w + effective_filter_w - in_w, 0)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        # 形态学膨胀：padding 区域填充负无穷大，使 max 操作忽略这些位置
        x = torch.nn.functional.pad(x, [pad_left, pad_right, pad_top, pad_bottom], value=float('-inf'))
    elif padding_mode == 'VALID':
        # VALID 模式不需要 padding（或使用指定 pads）
        if pads and sum(pads) > 0:
            x = torch.nn.functional.pad(x, [pads[2], pads[3], pads[0], pads[1]], value=float('-inf'))
        out_h = (in_h - effective_filter_h + stride_h) // stride_h
        out_w = (in_w - effective_filter_w + stride_w) // stride_w
    else:
        if pads and sum(pads) > 0:
            x = torch.nn.functional.pad(x, [pads[2], pads[3], pads[0], pads[1]], value=float('-inf'))
        out_h = (x.shape[2] - effective_filter_h + stride_h) // stride_h
        out_w = (x.shape[3] - effective_filter_w + stride_w) // stride_w
        if ceil_mode:
            out_h = (x.shape[2] - effective_filter_h + stride_h - 1) // stride_h + 1
            out_w = (x.shape[3] - effective_filter_w + stride_w - 1) // stride_w + 1

    # 形态学膨胀: 使用 unfold 获取 patches
    patches = torch.nn.functional.unfold(
        x,
        kernel_size=(effective_filter_h, effective_filter_w),
        dilation=(rate_h, rate_w),
        stride=(stride_h, stride_w)
    )

    # patches shape: [batch, channels * filter_h * filter_w, out_h * out_w]
    patches = patches.view(batch, channels, filter_h, filter_w, out_h, out_w)

    # 形态学膨胀：input_patch + filter，然后取最大值
    # filter shape: [C, H, W] -> expand to [batch, C, filter_h, filter_w, out_h, out_w]
    filter_expanded = filter.unsqueeze(0).unsqueeze(4).unsqueeze(5).expand(batch, -1, -1, -1, out_h, out_w)

    # 对每个 patch 位置，计算 input + filter，然后取最大值
    y = (patches + filter_expanded).max(dim=4)[0].max(dim=3)[0]

    if data_format == 'NHWC':
        y = y.permute(0, 2, 3, 1)  # NCHW -> NHWC

    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.randn(2, 64, 64, 64, dtype=torch.float16, device="npu")  # NHWC: [N, H, W, C]
filter = torch.randn(3, 3, 64, dtype=torch.float16, device="npu")   # [filter_h, filter_w, C]

y = cann_bench.dilation2_d(x, filter, strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding_mode='SAME', pads=[0, 0, 0, 0], ceil_mode=False, data_format='NHWC')
```
