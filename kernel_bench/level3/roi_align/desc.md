# ROIAlign 算子 API 描述

## 1. 算子简介

池化层，用于非均匀输入尺寸的特征图。

**主要应用场景**：
- 目标检测中对候选区域（Region of Interest）进行特征提取
- Faster R-CNN、Mask R-CNN 等两阶段检测框架中的特征对齐
- 实例分割中将不同大小的 ROI 映射到固定尺寸的特征表示

**算子特征**：
- 难度等级：L3（FusedComposite）
- 双输入（特征图 x 和 ROI 框 rois）单输出，支持双线性和最近邻两种插值模式

## 2. 算子定义

### 数学公式

$$
y = \text{roi\_align}(x, rois, \text{output\_size})
$$

对于每个 ROI，将其映射到输入特征图上的区域（通过 spatial_scale 缩放），然后将该区域划分为 outputHeight x outputWidth 个 bin，在每个 bin 内通过双线性插值（或最近邻插值）采样后进行平均池化，得到固定尺寸的输出。

## 3. 接口规范

### 算子原型

```python
ascend_bench.roi_align(Tensor x, Tensor rois, str mode, int outputHeight, int outputWidth, float spatial_scale, int sampling_ratio, bool aligned) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入特征图，shape 为 [B, C, H, W] |
| rois | Tensor | 必选 | ROI 框，shape 为 [numRois, 4] 或 [numRois, 5] |
| mode | str | 必选 | 插值模式（'bilinear': 双线性, 'nearest': 最近邻） |
| outputHeight | int | 必选 | 输出高度 |
| outputWidth | int | 必选 | 输出宽度 |
| spatial_scale | float | 必选 | 空间缩放因子（用于将 ROI 坐标映射到输入特征图尺寸） |
| sampling_ratio | int | -1 | 采样比率 |
| aligned | bool | false | 是否对齐 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | [numRois, C, outputHeight, outputWidth] | 与输入 x 相同 | 输出张量，ROI 对齐结果 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float32 | float32 |
| float16 | float16 |

### 规则与约束

- 输入特征图 x 的 shape 为 [B, C, H, W]，即 batch、通道、高、宽四维格式
- ROI 框 rois 的 shape 为 [numRois, 4] 或 [numRois, 5]，其中 5 列格式包含 batch_index
- x 和 rois 的 dtype 需一致
- mode 仅支持 'bilinear' 和 'nearest' 两种插值模式
- outputHeight 和 outputWidth 需为正整数
- spatial_scale 用于将 ROI 坐标从原图尺度映射到特征图尺度
- sampling_ratio 为 -1 或 0 时自动计算采样点数

## 4. 精度要求

计算结果与 PyTorch Golden 实现逐元素对比，需满足以下误差阈值：

| 数据类型 | 验证方式 | rtol | atol |
|---------|---------|------|------|
| float16 | 相对误差 | 1e-3 | 1e-3 |
| float32 | 相对误差 | 1e-4 | 1e-4 |

**对比公式**：

$$
|output - golden| \leq atol + rtol \times |golden|
$$

## 5. 标准 Golden 代码

```python
import torch

def roi_align(
    x: torch.Tensor, rois: torch.Tensor, pooled_height: int, pooled_width: int,
    spatial_scale: float = 1.0, sample_ratio: int = -1, aligned: bool = False
) -> torch.Tensor:
    """
    池化层，用于非均匀输入尺寸的特征图

    公式: y = roi_align(x, rois, output_size)

    Args:
        x: 输入特征图 [N, C, H, W]
        rois: ROI框 [K, 5] (batch_idx, x1, y1, x2, y2)
        pooled_height: 输出高度
        pooled_width: 输出宽度
        spatial_scale: 空间缩放因子
        sample_ratio: 采样比率
        aligned: 是否对齐

    Returns:
        输出张量 [K, C, pooled_height, pooled_width]
    """

    num_rois = rois.shape[0]
    channels = x.shape[1]

    output = torch.zeros(
        (num_rois, channels, pooled_height, pooled_width),
        dtype=x.dtype, device=x.device
    )

    for i in range(num_rois):
        roi = rois[i]
        batch_idx = int(roi[0])
        x1, y1, x2, y2 = roi[1:].tolist()

        x1 *= spatial_scale
        y1 *= spatial_scale
        x2 *= spatial_scale
        y2 *= spatial_scale

        roi_width = max(x2 - x1, 0.0)
        roi_height = max(y2 - y1, 0.0)

        bin_size_h = roi_height / pooled_height
        bin_size_w = roi_width / pooled_width

        if sample_ratio <= 0:
            n_sample = max(int(max(roi_width, roi_height) // pooled_height), 1)
        else:
            n_sample = sample_ratio

        roi_data = x[batch_idx:batch_idx + 1]

        for ph in range(pooled_height):
            for pw in range(pooled_width):
                y_start = y1 + ph * bin_size_h
                x_start = x1 + pw * bin_size_w
                y_end = y_start + bin_size_h
                x_end = x_start + bin_size_w

                ys = torch.linspace(y_start, y_end, n_sample, device=x.device)
                xs = torch.linspace(x_start, x_end, n_sample, device=x.device)
                grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')

                # Normalize to [-1, 1]
                H, W = roi_data.shape[2], roi_data.shape[3]
                norm_x = 2.0 * grid_x / (W - 1) - 1.0
                norm_y = 2.0 * grid_y / (H - 1) - 1.0
                grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)

                sampled = torch.nn.functional.grid_sample(
                    roi_data, grid, mode='bilinear', padding_mode='zeros', align_corners=not aligned
                )
                output[i, :, ph, pw] = sampled.mean(dim=[0, 2, 3])

    return output
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import ascend_bench

x = torch.randn(2, 256, 64, 64, dtype=torch.float32, device="npu")
rois = torch.tensor([[0, 10.0, 10.0, 50.0, 50.0], [1, 20.0, 20.0, 60.0, 60.0]], dtype=torch.float32, device="npu")
y = ascend_bench.roi_align(x, rois, "bilinear", 7, 7, 0.0625, 2, False)
```

### 性能基线参考

基于 cases.yaml 中 20 个测试用例，所有用例的 baseline_perf_us 均为 None，基线性能尚未测量。测试用例覆盖了不同的特征图大小（32x32 到 128x128）、不同的输出尺寸（1x1 到 14x14）、不同的 spatial_scale（0.015625 到 1.0）以及 float16、float32、bfloat16 等数据类型。

### 相关算子

- **ROIPooling**：对输入特征图按 ROI 进行最大池化，与 ROIAlign 类似但使用量化坐标而非插值
- **Conv2D**：二维卷积算子，常与 ROIAlign 配合用于特征提取网络
- **NMS**：非极大值抑制算子，在目标检测流水线中与 ROIAlign 协同使用
