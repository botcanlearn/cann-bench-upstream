# GridSampler3D 算子 API 描述

## 1. 算子简介

根据 grid 中坐标信息填充输出，实现 3D 空间的网格采样。

**主要应用场景**：
- 3D 医学图像处理中的空间变换与配准
- 3D 目标检测中的特征对齐与采样
- 体素数据的插值与重采样
- 空间变换网络（STN）中的可微分采样

**算子特征**：
- 难度等级：L3（FusedComposite）
- 双输入单输出，输入为 5D 张量 (N, C, D, H, W) 和采样网格 (N, D, H, W, 3)
- 支持多种插值模式和填充模式

## 2. 算子定义

### 数学公式

$$
y = \text{grid\_sample}(x, \text{grid})
$$

根据 grid 中指定的归一化坐标 $(\theta_d, \theta_h, \theta_w) \in [-1, 1]^3$，从输入张量 $x$ 中采样得到输出 $y$。

- **bilinear 模式**：使用三线性插值（trilinear interpolation），根据 8 个相邻体素加权计算采样值
- **nearest 模式**：取最近邻体素的值
- **align_corners**：控制坐标映射方式；为 true 时 $-1$ 和 $1$ 映射到角点像素中心，为 false 时映射到角点像素边界

## 3. 接口规范

### 算子原型

```python
cann_bench.grid_sampler_3d(Tensor x, Tensor grid, str interpolation_mode="bilinear", str padding_mode="zeros", bool align_corners=false) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入张量，形状为 (N, C, D, H, W) |
| grid | Tensor | 必选 | 采样网格，形状为 (N, D_out, H_out, W_out, 3) |
| interpolation_mode | str | "bilinear" | 插值模式（'bilinear': 双线性，'nearest': 最近邻，'bicubic': 双三次） |
| padding_mode | str | "zeros" | 填充模式（'zeros': 零填充，'border': 边界填充，'reflection': 反射填充） |
| align_corners | bool | false | 是否对齐角点 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | (N, C, D_out, H_out, W_out) | 与输入 x 相同 | 输出张量，采样结果 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |

### 规则与约束

- 输入 x 必须为 5D 张量，形状为 (N, C, D, H, W)
- grid 必须为 5D 张量，形状为 (N, D_out, H_out, W_out, 3)，最后一维为 3 表示 (d, h, w) 坐标
- x 和 grid 的 batch 维度 N 必须一致
- x 和 grid 的 dtype 必须一致
- grid 中的坐标值通常归一化到 [-1, 1] 范围
- interpolation_mode 可选 'bilinear'、'nearest'、'bicubic'
- padding_mode 可选 'zeros'、'border'、'reflection'

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

"""
GridSampler3D算子Torch Golden参考实现

根据grid中坐标信息填充输出
公式: y = grid_sample(x, grid)
"""
def grid_sampler_3d(
    x: torch.Tensor, grid: torch.Tensor, interpolation_mode: str = 'bilinear', padding_mode: str = 'zeros', align_corners: bool = False
) -> torch.Tensor:
    """
    根据grid中坐标信息填充输出
    
    公式: y = grid_sample(x, grid)
    
    Args:
        x: 输入张量
        grid: 采样网格
        interpolation_mode: 插值模式 ('bilinear': 双线性, 'nearest': 最近邻, 'bicubic': 双三次)
        padding_mode: 填充模式 ('zeros': 零填充, 'border': 边界填充, 'reflection': 反射填充)
        align_corners: 是否对齐角点
    
    Returns:
        输出张量，采样结果
    """

    return torch.nn.functional.grid_sample(
        x, grid, mode=interpolation_mode, padding_mode=padding_mode, align_corners=align_corners
    )
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.randn(2, 32, 16, 64, 64, dtype=torch.float16, device="npu")
grid = torch.rand(2, 8, 8, 8, 3, dtype=torch.float16, device="npu") * 2 - 1  # 归一化到 [-1, 1]
y = cann_bench.grid_sampler_3d(x, grid, interpolation_mode="bilinear", padding_mode="zeros", align_corners=True)

# nearest 插值 + border 填充
y = cann_bench.grid_sampler_3d(x, grid, interpolation_mode="nearest", padding_mode="border", align_corners=False)
```

### 性能基线参考

基于 cases.yaml 中 20 个测试用例，所有用例的 baseline_perf_us 均为 None，性能基线数据尚未测量。

### 相关算子

- **ResizeBilinearV2**：使用双线性插值调整图像大小，同为插值类算子
- **Scatter**：按索引更新张量，同涉及坐标映射操作
- **Gather**：按索引采集数据，与网格采样在概念上相关
