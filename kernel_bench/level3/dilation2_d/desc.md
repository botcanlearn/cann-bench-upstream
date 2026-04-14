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
y[b, y, x, c] = \max_{dy, dx} \left( x[b, y + \text{rates}[1] \cdot dy, x + \text{rates}[2] \cdot dx, c] \times \text{filter}[dy, dx, c] \right)
$$

对每个输出位置 $(b, y, x, c)$，在以 rates 确定的空洞采样窗口内，计算输入与结构元素逐元素乘积的最大值。

## 3. 接口规范

### 算子原型

```python
ascend_bench.dilation2_d(Tensor x, Tensor filter, int[] strides, int[] rates, str padding_mode, int[] pads, bool ceil_mode, str data_format) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入图像 |
| filter | Tensor | 必选 | 结构元素/卷积核 |
| strides | int[] | 必选 | 滑动窗口的步长 [1, stride_rows, stride_cols, 1] |
| rates | int[] | 必选 | 膨胀率 [1, rate_rows, rate_cols, 1]，用于空洞膨胀 |
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

计算结果与 PyTorch Golden 实现逐元素对比，需满足以下误差阈值：

| 数据类型 | 验证方式 | rtol | atol |
|---------|---------|------|------|
| float16 | 相对误差 | 1e-3 | 1e-3 |

**对比公式**：

$$
|output - golden| \leq atol + rtol \times |golden|
$$

## 5. 标准 Golden 代码

```python
import torch

"""
Dilation2D 算子 Torch Golden 参考实现

2D形态学膨胀操作，使用最大池化在局部邻域内获取最大值
公式: y[b, y, x, c] = max_{dy,dx} x[b, y + rates[1]*dy, x + rates[2]*dx, c] * filter[dy, dx, c]
"""
def dilation2_d(
    x: torch.Tensor, kernel_size: list, strides: list,
    pads: list = [0, 0, 0, 0], dilations: list = [1, 1],
    padding_mode: str = 'SAME', ceil_mode: bool = False,
    data_format: str = 'NHWC'
) -> torch.Tensor:
    """
    2D形态学膨胀操作，使用最大池化在局部邻域内获取最大值

    公式: y[b, y, x, c] = max_{dy,dx} x[b, y + rates[1]*dy, x + rates[2]*dx, c] * filter[dy, dx, c]

    Args:
        x: 输入图像
        kernel_size: 卷积核尺寸 [height, width]
        strides: 步长 [stride_h, stride_w]
        pads: 填充值 [pad_top, pad_bottom, pad_left, pad_right]
        dilations: 膨胀率 [dilation_h, dilation_w]
        padding_mode: 填充模式：'SAME' 或 'VALID'
        ceil_mode: 是否向上取整计算输出尺寸
        data_format: 数据格式，如 'NHWC'

    Returns:
        膨胀后的图像
    """

    if data_format == 'NHWC':
        x = x.permute(0, 3, 1, 2)

    batch, channels, in_h, in_w = x.shape
    filter_h, filter_w = kernel_size[0], kernel_size[1]
    stride_h, stride_w = strides[0], strides[1]
    rate_h, rate_w = dilations[0], dilations[1]

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
        x = torch.nn.functional.pad(x, [pad_left, pad_right, pad_top, pad_bottom])
    elif padding_mode == 'VALID':
        if pads and sum(pads) > 0:
            x = torch.nn.functional.pad(x, [pads[2], pads[3], pads[0], pads[1]])
        out_h = (in_h - effective_filter_h + stride_h) // stride_h
        out_w = (in_w - effective_filter_w + stride_w) // stride_w
    else:
        if pads and sum(pads) > 0:
            x = torch.nn.functional.pad(x, [pads[2], pads[3], pads[0], pads[1]])
        out_h = (x.shape[2] - effective_filter_h + stride_h) // stride_h
        out_w = (x.shape[3] - effective_filter_w + stride_w) // stride_w
        if ceil_mode:
            out_h = (x.shape[2] - effective_filter_h + stride_h - 1) // stride_h + 1
            out_w = (x.shape[3] - effective_filter_w + stride_w - 1) // stride_w + 1

    # 形态学膨胀: 使用 unfold 获取 patches，然后取最大值
    patches = torch.nn.functional.unfold(
        x,
        kernel_size=(effective_filter_h, effective_filter_w),
        dilation=(rate_h, rate_w),
        stride=(stride_h, stride_w)
    )

    patches = patches.view(batch, channels, filter_h, filter_w, out_h, out_w)

    # 形态学膨胀：对每个位置取 filter 全为1情况下的最大值
    y = patches.max(dim=4)[0].max(dim=3)[0]

    if data_format == 'NHWC':
        y = y.permute(0, 2, 3, 1)

    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import ascend_bench

x = torch.randn(2, 32, 64, 64, dtype=torch.float16, device="npu")  # NHWC: [N, H, W, C]
filter = torch.randn(3, 3, 64, dtype=torch.float16, device="npu")   # [filter_h, filter_w, C]

y = ascend_bench.dilation2_d(x, filter, strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding_mode='SAME', pads=[0, 0, 0, 0], ceil_mode=False, data_format='NHWC')
```

### 性能基线参考

基于 cases.yaml 中 20 个测试用例，所有用例的 baseline_perf_us 均为 None，基线性能尚未测量。测试用例覆盖 3x3、5x5、7x7 等不同卷积核大小，膨胀率从 1 到 3，步长从 1 到 2，包含非对齐质数 shape（如 [3, 7, 97, 103]）、零值输入、特殊值范围（inf）和非对齐 batch（2049）等边界场景。

### 相关算子

- **Conv2D**：2D 卷积运算，与 Dilation2D 结构类似但使用求和代替取最大值
- **DepthwiseConv2D**：深度可分离卷积，同样使用滑动窗口处理2D输入
- **RoiPooling**：区域池化操作，涉及空间维度的池化计算
