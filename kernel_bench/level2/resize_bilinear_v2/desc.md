# ResizeBilinearV2 算子 API 描述

## 1. 算子简介

使用双线性插值调整图像大小。

**主要应用场景**：
- 图像预处理中的缩放与分辨率调整
- 特征金字塔网络（FPN）中的上采样操作
- 语义分割中的特征图恢复到原始分辨率
- 目标检测中不同尺度特征的对齐

**算子特征**：
- 难度等级：L2（FusedComposite）
- 单输入单输出，输入为 4D 张量 (N, C, H, W)，输出空间维度由 output_size 或 scale_factor 指定

## 2. 算子定义

### 数学公式

$$
y = \text{resize\_bilinear}(x, \text{size})
$$

对输入张量的空间维度 (H, W) 进行双线性插值缩放。对于输出位置 $(i, j)$，根据其在输入空间中的映射坐标 $(h, w)$，利用周围 4 个最近邻像素的值进行加权平均：

$$
y(i, j) = (1-\alpha)(1-\beta) \cdot x(\lfloor h \rfloor, \lfloor w \rfloor) + \alpha(1-\beta) \cdot x(\lceil h \rceil, \lfloor w \rfloor) + (1-\alpha)\beta \cdot x(\lfloor h \rfloor, \lceil w \rceil) + \alpha\beta \cdot x(\lceil h \rceil, \lceil w \rceil)
$$

其中 $\alpha = h - \lfloor h \rfloor$，$\beta = w - \lfloor w \rfloor$。

- **align_corners=true**：输入输出的角点像素对齐，坐标映射为 $h = i \times \frac{H_{in}-1}{H_{out}-1}$
- **align_corners=false**：坐标映射为 $h = (i + 0.5) \times \frac{H_{in}}{H_{out}} - 0.5$

## 3. 接口规范

### 算子原型

```python
cann_bench.resize_bilinear_v2(Tensor x, int[] output_size, bool align_corners=false, float[] scale_factor=null) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入张量，形状为 (N, C, H, W) |
| output_size | int[] | 必选 | 输出尺寸 [output_height, output_width] |
| align_corners | bool | false | 是否对齐角点 |
| scale_factor | float[] | null | 缩放因子 [scale_height, scale_width]，与 output_size 互斥 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | (N, C, H_out, W_out) | 与输入 x 相同 | 输出张量，调整大小后的结果 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

### 规则与约束

- 输入 x 必须为 4D 张量，形状为 (N, C, H, W)
- output_size 和 scale_factor 互斥，两者不能同时指定
- output_size 为 [output_height, output_width]，指定输出空间维度大小
- scale_factor 为 [scale_height, scale_width]，指定缩放比例
- 输出 dtype 与输入 dtype 一致
- 支持上采样（输出大于输入）和下采样（输出小于输入）

## 4. 精度要求

计算结果与 PyTorch Golden 实现逐元素对比，需满足以下误差阈值：

| 数据类型 | 验证方式 | rtol | atol |
|---------|---------|------|------|
| float16 | 相对误差 | 1e-3 | 1e-3 |
| float32 | 相对误差 | 1e-4 | 1e-4 |
| bfloat16 | 相对误差 | 4e-3 | 4e-3 |

**对比公式**：

$$
|output - golden| \leq atol + rtol \times |golden|
$$

## 5. 标准 Golden 代码

```python
import torch
from typing import List, Optional

"""
ResizeBilinearV2 算子 Torch Golden 参考实现

使用双线性插值调整图像大小
公式：y = resize_bilinear(x, size)
"""
def resize_bilinear_v2(
    x: torch.Tensor,
    output_size: Optional[List[int]] = None,
    align_corners: bool = False,
    scale_factor: Optional[List[float]] = None
) -> torch.Tensor:
    """
    使用双线性插值调整图像大小

    Args:
        x: 输入张量，形状为 (N, C, H, W)
        output_size: 输出尺寸 [output_height, output_width]
        align_corners: 是否对齐角点
        scale_factor: 缩放因子 [scale_height, scale_width]，与 output_size 互斥

    Returns:
        输出张量，调整大小后的结果
    """
    # 使用 PyTorch 的 interpolate 实现双线性插值
    y = torch.nn.functional.interpolate(
        x,
        size=output_size,
        scale_factor=scale_factor[0] if scale_factor and len(scale_factor) == 1 else scale_factor,
        mode='bilinear',
        align_corners=align_corners
    )
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.randn(2, 8, 512, 512, dtype=torch.float16, device="npu")
y = cann_bench.resize_bilinear_v2(x, output_size=[256, 256], align_corners=False)  # 下采样

x = torch.randn(4, 4, 64, 64, dtype=torch.float32, device="npu")
y = cann_bench.resize_bilinear_v2(x, output_size=[128, 128], align_corners=True)  # 上采样 + 角点对齐

x = torch.randn(1, 16, 128, 128, dtype=torch.bfloat16, device="npu")
y = cann_bench.resize_bilinear_v2(x, output_size=[256, 256])  # bfloat16 上采样
```

### 性能基线参考

基于 cases.yaml 中 20 个测试用例，当前所有用例的 baseline_perf_us 均为 0，性能基线数据待补充。

### 相关算子

- **GridSampler3D**：同为插值类算子，支持更灵活的采样坐标映射
- **Gather**：按索引采集数据，与插值采样在数据访问模式上相关
- **DynamicQuant**：涉及数据精度转换，同为数据变换类算子
