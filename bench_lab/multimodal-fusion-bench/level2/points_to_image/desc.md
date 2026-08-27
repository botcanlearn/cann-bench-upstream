# PointsToImage 算子 API 描述

## 1. 算子简介

PointsToImage 算子用于将三维点通过相机投影矩阵投影到二维图像平面，输出对应的像素坐标。该算子是相机-激光雷达多模态融合中的基础几何原语，可用于点云投影、体素中心投影、3D 检测框角点投影、BEV 网格投影以及 frustum 构造前处理。

**主要应用场景**：
- 将 LiDAR 点投影到图像平面，用于图像语义查询和 PointPainting
- 将 3D box corner 投影到图像平面，用于 2D/3D 候选关联
- 将体素中心或 BEV 网格中心投影到相机视角
- 构造 frustum、图像 ROI 点集和跨模态几何对应关系

**算子特征**：
- 难度等级：L1（Geometry）
- 双输入单输出，输出 shape 与输入点集前缀维度一致，最后一维由 3 变为 2
- 支持单个投影矩阵 `(3, 4)` 和 batch 投影矩阵 `(B, 3, 4)`
- 支持任意点集前缀维度，例如 `(N, 3)`、`(B, N, 3)`、`(B, K, N, 3)`

## 2. 算子定义

### 数学公式

给定三维点 $p=(x,y,z)$ 和投影矩阵 $P \in R^{3 \times 4}$，先构造齐次坐标：

$$
p_h = [x, y, z, 1]^T
$$

然后进行投影：

$$
[x_i, y_i, z_i]^T = P \cdot p_h
$$

最终像素坐标为：

$$
u = \frac{x_i}{max(z_i, eps)}, \quad v = \frac{y_i}{max(z_i, eps)}
$$

### 特殊情况

| 条件 | 输出 shape |
|------|------------|
| points_3d 为 `(N, 3)`，proj_matrix 为 `(3, 4)` | `(N, 2)` |
| points_3d 为 `(B, N, 3)`，proj_matrix 为 `(3, 4)` | `(B, N, 2)` |
| points_3d 为 `(B, N, 3)`，proj_matrix 为 `(B, 3, 4)` | `(B, N, 2)` |
| points_3d 为 `(B, K, N, 3)`，proj_matrix 为 `(B, 3, 4)` | `(B, K, N, 2)` |

## 3. 接口规范

### 算子原型

```python
cann_bench.points_to_image(Tensor points_3d, Tensor proj_matrix, float eps=1e-6) -> Tensor pixel_coords
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| points_3d | Tensor | 必选 | 三维点坐标，shape 为 `(..., 3)`，最后一维为 `(x, y, z)` |
| proj_matrix | Tensor | 必选 | 相机投影矩阵，shape 为 `(3, 4)` 或 `(B, 3, 4)` |
| eps | float | 1e-6 | 深度分母最小值，用于避免除零 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| pixel_coords | `(..., 2)` | 与 `points_3d` 相同 | 二维像素坐标，最后一维为 `(u, v)` |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |
| float64 | float64 |

### 规则与约束

- `points_3d` 最后一维必须为 3，表示 `(x, y, z)`
- `proj_matrix` 必须为 `(3, 4)` 或 `(B, 3, 4)`
- 当 `proj_matrix` 为 batch 格式时，`points_3d` 第 0 维必须与 `proj_matrix` 的 batch 维相同
- `proj_matrix` 会被转换到 `points_3d` 的 device 和 dtype 后参与计算
- 输出不进行图像边界裁剪，不判断点是否在图像内

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `points_3d` 维度数 | 2 ~ 4 | cases.csv 实测 `(N,3)`、`(B,N,3)`、`(B,K,N,3)` |
| 点数量 | 1 ~ 65536 | cases.csv 实测总点数 1 ~ 65536 |
| `B` | 1 ~ 8 | batch 投影矩阵实测 1 ~ 8 |
| `K` | 1 ~ 16 | 高维点集前缀维度实测 1 ~ 16 |
| `eps` | 1e-8 ~ 1e-3 | cases.csv 实测 1e-8、1e-6、1e-4、1e-3 |

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

| 数据类型 | FLOAT16 | FLOAT32 | FLOAT64 |
|----------|---------|---------|---------|
| **通过阈值(Threshold)** | 2^-10 | 2^-13 | 2^-20 |

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过。

## 5. 标准 Golden 代码

fp16 输入时升精到 fp32 计算，结果转回 fp16 输出。

```python
import torch

def points_to_image(
    points_3d: torch.Tensor,
    proj_matrix: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    orig_shape = points_3d.shape[:-1]
    device = points_3d.device
    input_dtype = points_3d.dtype

    if input_dtype == torch.float16:
        compute_dtype = torch.float32
    else:
        compute_dtype = input_dtype

    points_compute = points_3d.to(compute_dtype)
    P = proj_matrix.to(device=device, dtype=compute_dtype)

    if P.dim() == 3:
        batch_size = P.shape[0]
        pts = points_compute.reshape(batch_size, -1, 3)
        ones = torch.ones((batch_size, pts.shape[1], 1), device=device, dtype=compute_dtype)
        pts_h = torch.cat([pts, ones], dim=-1)
        pixels_h = pts_h @ P.transpose(-1, -2)
        depth = pixels_h[..., 2:3].clamp_min(float(eps))
        pixels = pixels_h[..., :2] / depth
        pixels = pixels.reshape(orig_shape + (2,))
    else:
        pts = points_compute.reshape(-1, 3)
        ones = torch.ones((pts.shape[0], 1), device=device, dtype=compute_dtype)
        pts_h = torch.cat([pts, ones], dim=1)
        pixels_h = pts_h @ P.t()
        depth = pixels_h[:, 2:3].clamp_min(float(eps))
        pixels = pixels_h[:, :2] / depth
        pixels = pixels.reshape(orig_shape + (2,))

    if input_dtype == torch.float16:
        return pixels.to(input_dtype)
    return pixels
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

points_3d = torch.randn(4096, 3, dtype=torch.float32, device="npu")
proj_matrix = torch.randn(3, 4, dtype=torch.float32, device="npu")
pixel_coords = cann_bench.points_to_image(points_3d, proj_matrix, eps=1e-6)
```
