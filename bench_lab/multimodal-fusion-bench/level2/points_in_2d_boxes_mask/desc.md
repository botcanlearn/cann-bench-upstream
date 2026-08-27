# PointsIn2DBoxesMask 算子 API 描述

## 1. 算子简介

PointsIn2DBoxesMask 算子用于判断二维图像平面上的点是否落入二维检测框内，输出布尔掩码。该算子是多模态感知中的通用几何过滤原语，可用于 frustum 构造、PointPainting 点级语义查询、ROI 预筛选以及 2D-3D 候选关联等场景。

**主要应用场景**：
- Frustum 构造中筛选落入 2D 检测框的激光雷达投影点
- PointPainting 中判断投影点是否命中图像语义区域或实例框
- 2D/3D 检测结果融合前的候选点、候选框预过滤
- 图像 ROI 内点集统计、掩码构造和关联匹配

**算子特征**：
- 难度等级：L1（Geometry）
- 双输入单输出，输出为 bool 类型
- 支持非 batch 和 batch 两类输入
- 支持单框、多框、每 batch 单框、每 batch 多框四种框输入格式

## 2. 算子定义

### 数学公式

给定二维点 $p=(u,v)$ 和二维框 $b=[x_1,y_1,x_2,y_2]$，点在框内的判断公式为：

$$
mask = (u \ge x_1) \land (u < x_2) \land (v \ge y_1) \land (v < y_2)
$$

当 `use_image_bounds=True` 时，还需要满足图像边界约束：

$$
mask = mask \land (u \ge 0) \land (u < image\_width) \land (v \ge 0) \land (v < image\_height)
$$

### 特殊情况

| 条件 | 输出 shape |
|------|------------|
| points_2d 为 (N, 2)，boxes_2d 为 (4) | (N) |
| points_2d 为 (N, 2)，boxes_2d 为 (K, 4) | (K, N) |
| points_2d 为 (B, N, 2)，boxes_2d 为 (4) 或 (B, 4) | (B, N) |
| points_2d 为 (B, N, 2)，boxes_2d 为 (B, K, 4) | (B, K, N) |

## 3. 接口规范

### 算子原型

```python
cann_bench.points_in_2d_boxes_mask(Tensor points_2d, Tensor boxes_2d, int image_width, int image_height, bool use_image_bounds=True) -> Tensor mask
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| points_2d | Tensor | 必选 | 二维点坐标，shape 为 (N, 2) 或 (B, N, 2)，最后一维为 (u, v) |
| boxes_2d | Tensor | 必选 | 二维框，shape 为 (4)、(K, 4)、(B, 4) 或 (B, K, 4)，格式为 [x1, y1, x2, y2] |
| image_width | int | 必选 | 图像宽度 |
| image_height | int | 必选 | 图像高度 |
| use_image_bounds | bool | True | 是否要求点同时位于图像边界内 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| mask | (N)、(K, N)、(B, N) 或 (B, K, N) | bool | 点是否位于对应二维框内的布尔掩码 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | bool |
| float32 | bool |
| float64 | bool |
| int32 | bool |
| int64 | bool |

### 规则与约束

- `points_2d` 最后一维必须为 2，表示 `(u, v)`
- `boxes_2d` 最后一维必须为 4，表示 `[x1, y1, x2, y2]`
- 框的左上边界为闭区间，右下边界为开区间，即 `x1 <= u < x2`、`y1 <= v < y2`
- 当 `use_image_bounds=True` 时，越出图像范围的点即使落在框坐标范围内也输出 `False`
- 当 `use_image_bounds=False` 时，只执行点与框的坐标比较，不额外过滤图像外点

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `points_2d` 维度数 | 2 或 3 | 非 batch: (N, 2)，batch: (B, N, 2) |
| `boxes_2d` 维度数 | 1、2 或 3 | 支持单框、多框、batch 单框、batch 多框 |
| `N` | 1 ~ 65536 | cases.csv 实测 1 ~ 65536 |
| `K` | 1 ~ 64 | cases.csv 实测 1 ~ 64 |
| `B` | 1 ~ 4 | cases.csv 实测 1 ~ 4 |
| `image_width` | 1 ~ 3840 | cases.csv 实测 1 ~ 3840 |
| `image_height` | 1 ~ 2160 | cases.csv 实测 1 ~ 2160 |

## 4. 精度要求

该算子输出为 bool 掩码，采用精确一致性校验：

```text
actual == golden
```

当所有输出元素均与 Golden 结果一致时判定为通过。

## 5. 标准 Golden 代码

```python
import torch

def points_in_2d_boxes_mask(
    points_2d: torch.Tensor,
    boxes_2d: torch.Tensor,
    image_width: int,
    image_height: int,
    use_image_bounds: bool = True,
) -> torch.Tensor:
    single_points = points_2d.dim() == 2
    if single_points:
        pts = points_2d.unsqueeze(0)
    else:
        pts = points_2d

    B = pts.shape[0]
    u = pts[..., 0]
    v = pts[..., 1]
    boxes = boxes_2d.to(device=pts.device, dtype=pts.dtype)
    single_box = boxes.dim() == 1

    if single_box:
        boxes = boxes.view(1, 1, 4).expand(B, 1, 4)
    elif boxes.dim() == 2:
        if boxes.shape[0] == B and not single_points:
            boxes = boxes.view(B, 1, 4)
            single_box = True
        else:
            boxes = boxes.unsqueeze(0).expand(B, boxes.shape[0], 4)
    elif boxes.dim() == 3:
        if boxes.shape[0] != B:
            raise ValueError("batched boxes_2d must share batch size with points_2d")
    else:
        raise ValueError("boxes_2d must have shape (4), (K,4), (B,4), or (B,K,4)")

    x1 = boxes[..., 0].unsqueeze(-1)
    y1 = boxes[..., 1].unsqueeze(-1)
    x2 = boxes[..., 2].unsqueeze(-1)
    y2 = boxes[..., 3].unsqueeze(-1)

    uu = u.unsqueeze(1)
    vv = v.unsqueeze(1)
    mask = (uu >= x1) & (uu < x2) & (vv >= y1) & (vv < y2)

    if use_image_bounds:
        in_img = (uu >= 0) & (uu < image_width) & (vv >= 0) & (vv < image_height)
        mask = mask & in_img

    if single_box:
        mask = mask.squeeze(1)
    if single_points:
        mask = mask.squeeze(0)
    return mask
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

points_2d = torch.tensor([[100.0, 80.0], [300.0, 200.0]], device="npu")
boxes_2d = torch.tensor([[50.0, 50.0, 200.0, 150.0]], device="npu")
mask = cann_bench.points_in_2d_boxes_mask(
    points_2d, boxes_2d, image_width=640, image_height=480, use_image_bounds=True
)
```
