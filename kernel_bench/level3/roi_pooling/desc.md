# ROIPooling 算子 API 描述

## 1. 算子简介

对输入特征图按 ROI 进行最大池化。

**主要应用场景**：
- 目标检测中对候选区域进行固定尺寸的特征提取
- Fast R-CNN、Faster R-CNN 等检测框架中的 ROI 特征池化
- 将不同大小的感兴趣区域映射为统一尺寸的特征表示

**算子特征**：
- 难度等级：L3（Reduction）
- 双输入（特征图 x 和 ROI 框 rois）单输出，对每个 ROI 区域执行最大池化

## 2. 算子定义

### 数学公式

$$
y = \text{roi\_pool}(x, rois, \text{output\_size})
$$

对于每个 ROI，将其通过 spatial_scale 映射到输入特征图上的区域，然后将该区域划分为 pooled_h x pooled_w 个 bin，在每个 bin 内取最大值，得到固定尺寸的输出。

## 3. 接口规范

### 算子原型

```python
ascend_bench.roi_pooling(Tensor x, Tensor rois, int pooled_h, int pooled_w, float spatial_scale) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入特征图，shape 为 [N, C, H, W] |
| rois | Tensor | 必选 | ROI 框，shape 为 [num_rois, 5] |
| pooled_h | int | 必选 | 池化后高度 |
| pooled_w | int | 必选 | 池化后宽度 |
| spatial_scale | float | 必选 | 空间缩放因子（用于将 ROI 坐标映射到输入特征图尺寸） |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | [num_rois, C, pooled_h, pooled_w] | 与输入 x 相同 | 输出张量，ROI 池化结果 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float32 | float32 |
| float16 | float16 |

### 规则与约束

- 输入特征图 x 的 shape 为 [N, C, H, W]，即 batch、通道、高、宽四维格式
- ROI 框 rois 的 shape 为 [num_rois, 5]，每行格式为 [batch_index, x1, y1, x2, y2]
- x 和 rois 的 dtype 需一致
- pooled_h 和 pooled_w 需为正整数
- spatial_scale 用于将 ROI 坐标从原图尺度映射到特征图尺度

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

def roi_pooling(
    x: torch.Tensor, rois: torch.Tensor, pooled_h: int, pooled_w: int,
    spatial_scale: float = 1.0
) -> torch.Tensor:
    """
    对输入特征图按ROI进行最大池化

    公式: y = roi_pool(x, rois, output_size)

    Args:
        x: 输入特征图 [N, C, H, W]
        rois: ROI框 [K, 5] (batch_idx, x1, y1, x2, y2)
        pooled_h: 池化后高度
        pooled_w: 池化后宽度
        spatial_scale: 空间缩放因子

    Returns:
        输出张量 [K, C, pooled_h, pooled_w]
    """

    num_rois = rois.shape[0]
    channels = x.shape[1]

    output = torch.zeros(
        (num_rois, channels, pooled_h, pooled_w),
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

        bin_size_h = roi_height / pooled_h
        bin_size_w = roi_width / pooled_w

        roi_data = x[batch_idx:batch_idx + 1]

        for ph in range(pooled_h):
            for pw in range(pooled_w):
                y_start = int(y1 + ph * bin_size_h)
                x_start = int(x1 + pw * bin_size_w)
                y_end = int(y1 + (ph + 1) * bin_size_h)
                x_end = int(x1 + (pw + 1) * bin_size_w)

                y_start = max(y_start, 0)
                x_start = max(x_start, 0)
                y_end = min(y_end, roi_data.shape[2])
                x_end = min(x_end, roi_data.shape[3])

                if y_end > y_start and x_end > x_start:
                    output[i, :, ph, pw] = roi_data[:, :, y_start:y_end, x_start:x_end].max(dim=2)[0].max(dim=2)[0]

    return output
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import ascend_bench

x = torch.randn(2, 256, 64, 64, dtype=torch.float32, device="npu")
rois = torch.tensor([[0, 10.0, 10.0, 50.0, 50.0], [1, 20.0, 20.0, 60.0, 60.0]], dtype=torch.float32, device="npu")
y = ascend_bench.roi_pooling(x, rois, 7, 7, 0.0625)
```

### 性能基线参考

基于 cases.yaml 中 20 个测试用例，所有用例的 baseline_perf_us 均为 None，基线性能尚未测量。测试用例覆盖了不同的特征图大小（32x32 到 128x128）、不同的输出尺寸（1x1 到 14x14）、不同的 spatial_scale（0.015625 到 1.0）以及 float16、float32、bfloat16 等数据类型。

### 相关算子

- **ROIAlign**：与 ROIPooling 类似但使用双线性插值代替量化坐标，精度更高
- **Conv2D**：二维卷积算子，常与 ROIPooling 配合用于特征提取网络
- **NMS**：非极大值抑制算子，在目标检测流水线中与 ROIPooling 协同使用
