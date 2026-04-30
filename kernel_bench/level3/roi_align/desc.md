# ROIAlign 算子 API 描述

## 1. 算子简介

池化层，用于非均匀输入尺寸的特征图。

**主要应用场景**：
- 目标检测中对候选区域（Region of Interest）进行特征提取
- Faster R-CNN、Mask R-CNN 等两阶段检测框架中的特征对齐
- 实例分割中将不同大小的 ROI 映射到固定尺寸的特征表示

**算子特征**：
- 难度等级：L3（FusedComposite）
- 双输入（特征图 x 和 ROI 框 rois）单输出，支持双线性插值模式

## 2. 算子定义

### 数学公式

$$
y = \text{roi\_align}(x, \text{boxes}, \text{output\_size})
$$

对于每个 box，将其映射到输入特征图上的区域（通过 spatial_scale 缩放），然后将该区域划分为 outputHeight x outputWidth 个 bin，在每个 bin 内通过双线性插值采样后进行平均池化，得到固定尺寸的输出。

## 3. 接口规范

### 算子原型

```python
cann_bench.roi_align(Tensor x, Tensor boxes, int outputHeight, int outputWidth, float spatial_scale, int sampling_ratio, bool aligned) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入特征图，shape 为 [B, C, H, W] |
| boxes | Tensor | 必选 | ROI 框，shape 为 [numBoxes, 5] (batch_idx, x1, y1, x2, y2) |
| outputHeight | int | 必选 | 输出高度 |
| outputWidth | int | 必选 | 输出宽度 |
| spatial_scale | float | 必选 | 空间缩放因子（用于将 boxes 坐标映射到输入特征图尺寸） |
| sampling_ratio | int | -1 | 采样比率 (-1 或 0 时自动计算) |
| aligned | bool | false | 是否对齐 (aligned=True 时 boxes 坐标偏移 -0.5 像素) |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | [numBoxes, C, outputHeight, outputWidth] | 与输入 x 相同 | 输出张量，boxes 对齐结果 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float32 | float32 |
| float16 | float16 |

### 规则与约束

- 输入特征图 x 的 shape 为 [B, C, H, W]，即 batch、通道、高、宽四维格式
- ROI 框 boxes 的 shape 为 [numBoxes, 5]，其中 5 列格式为 (batch_idx, x1, y1, x2, y2)
- x 和 boxes 的 dtype 需一致
- outputHeight 和 outputWidth 需为正整数
- spatial_scale 用于将 ROI 坐标从原图尺度映射到特征图尺度
- sampling_ratio 为 -1 或 0 时自动计算采样点数

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

> 优先使用 `torchvision.ops.roi_align`，不可用时使用纯 Python fallback。

```python
import torch

try:
    from torchvision.ops import roi_align as _tv_roi_align
    HAS_TORCHVISION = True
except Exception:
    HAS_TORCHVISION = False


def _bilinear_interpolate(input, roi_batch_ind, y, x, ymask, xmask):
    _, channels, height, width = input.size()
    y = y.clamp(min=0); x = x.clamp(min=0)
    y_low = y.int(); x_low = x.int()
    y_high = torch.where(y_low >= height - 1, height - 1, y_low + 1)
    y_low = torch.where(y_low >= height - 1, height - 1, y_low)
    y = torch.where(y_low >= height - 1, y.to(input.dtype), y)
    x_high = torch.where(x_low >= width - 1, width - 1, x_low + 1)
    x_low = torch.where(x_low >= width - 1, width - 1, x_low)
    x = torch.where(x_low >= width - 1, x.to(input.dtype), x)
    ly = y - y_low; lx = x - x_low; hy = 1.0 - ly; hx = 1.0 - lx
    def masked_index(y_idx, x_idx):
        if ymask is not None:
            y_idx = torch.where(ymask[:, None, :], y_idx, 0)
            x_idx = torch.where(xmask[:, None, :], x_idx, 0)
        return input[roi_batch_ind[:, None, None, None, None, None],
                     torch.arange(channels, device=input.device)[None, :, None, None, None, None],
                     y_idx[:, None, :, None, :, None], x_idx[:, None, None, :, None, :]]
    v1 = masked_index(y_low, x_low); v2 = masked_index(y_low, x_high)
    v3 = masked_index(y_high, x_low); v4 = masked_index(y_high, x_high)
    def outer_prod(y_t, x_t):
        return y_t[:, None, :, None, :, None] * x_t[:, None, None, :, None, :]
    return outer_prod(hy, hx)*v1 + outer_prod(hy, lx)*v2 + outer_prod(ly, hx)*v3 + outer_prod(ly, lx)*v4


def _roi_align_fallback(x, boxes, pooled_height, pooled_width, spatial_scale, sample_ratio, aligned):
    orig_dtype = x.dtype
    x_fp32 = x.float(); boxes_fp32 = boxes.float()
    _, channels, height, width = x_fp32.size()
    roi_batch_ind = boxes_fp32[:, 0].long()
    offset = 0.5 if aligned else 0.0
    roi_start_w = boxes_fp32[:, 1] * spatial_scale - offset
    roi_start_h = boxes_fp32[:, 2] * spatial_scale - offset
    roi_end_w = boxes_fp32[:, 3] * spatial_scale - offset
    roi_end_h = boxes_fp32[:, 4] * spatial_scale - offset
    roi_width = roi_end_w - roi_start_w; roi_height = roi_end_h - roi_start_h
    if not aligned:
        roi_width = roi_width.clamp(min=1.0); roi_height = roi_height.clamp(min=1.0)
    bin_size_h = roi_height / pooled_height; bin_size_w = roi_width / pooled_width
    exact_sampling = sample_ratio > 0
    roi_bin_grid_h = sample_ratio if exact_sampling else torch.ceil(roi_height / pooled_height)
    roi_bin_grid_w = sample_ratio if exact_sampling else torch.ceil(roi_width / pooled_width)
    if exact_sampling:
        count = max(roi_bin_grid_h * roi_bin_grid_w, 1)
        iy = torch.arange(roi_bin_grid_h, device=x.device); ix = torch.arange(roi_bin_grid_w, device=x.device)
        ymask = None; xmask = None
    else:
        count = torch.clamp(roi_bin_grid_h * roi_bin_grid_w, min=1)
        iy = torch.arange(height, device=x.device); ix = torch.arange(width, device=x.device)
        ymask = iy[None, :] < roi_bin_grid_h[:, None]; xmask = ix[None, :] < roi_bin_grid_w[:, None]
    def from_K(t): return t[:, None, None]
    y = from_K(roi_start_h) + torch.arange(pooled_height, device=x.device)[None, :, None] * from_K(bin_size_h) + (iy[None, None, :] + 0.5).to(x_fp32.dtype) * from_K(bin_size_h / roi_bin_grid_h)
    x_pos = from_K(roi_start_w) + torch.arange(pooled_width, device=x.device)[None, :, None] * from_K(bin_size_w) + (ix[None, None, :] + 0.5).to(x_fp32.dtype) * from_K(bin_size_w / roi_bin_grid_w)
    val = _bilinear_interpolate(x_fp32, roi_batch_ind, y, x_pos, ymask, xmask)
    if not exact_sampling:
        val = torch.where(ymask[:, None, None, None, :, None], val, 0)
        val = torch.where(xmask[:, None, None, None, None, :], val, 0)
    output = val.sum((-1, -2))
    if isinstance(count, torch.Tensor): output = output / count[:, None, None, None]
    else: output = output / count
    return output.to(orig_dtype)


def roi_align(
    x: torch.Tensor, boxes: torch.Tensor, pooled_height: int, pooled_width: int,
    spatial_scale: float = 1.0, sample_ratio: int = -1, aligned: bool = False,
) -> torch.Tensor:
    """
    池化层，用于非均匀输入尺寸的特征图
    公式: y = roi_align(x, boxes, output_size)
    """
    if HAS_TORCHVISION:
        return _tv_roi_align(x, boxes, (pooled_height, pooled_width),
            spatial_scale=spatial_scale, sampling_ratio=sample_ratio, aligned=aligned)
    return _roi_align_fallback(x, boxes, pooled_height, pooled_width,
        spatial_scale, sample_ratio, aligned)
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.randn(2, 256, 64, 64, dtype=torch.float32, device="npu")
boxes = torch.tensor([[0, 10.0, 10.0, 50.0, 50.0], [1, 20.0, 20.0, 60.0, 60.0]], dtype=torch.float32, device="npu")
y = cann_bench.roi_align(x, boxes, 7, 7, 0.0625, 2, False)
```
