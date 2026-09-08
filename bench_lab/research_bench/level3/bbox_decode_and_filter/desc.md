# BboxDecodeAndFilter 算子 API 描述

## 1. 算子简介

`BboxDecodeAndFilter` 将 anchor boxes 与 bbox deltas 解码为预测框，并输出按 score threshold 过滤的 mask，用于补充 research benchmark 中的模型前后处理与表格特征路径覆盖。

**算子特征**：

- 难度等级：L3（Vision）
- anchors/deltas: [N,4], scores: [N], outputs decoded_boxes [N,4], keep_mask [N]
- 输出 shape 固定，便于精度比较

## 2. 算子定义

```text
decoded = apply_deltas(anchors,deltas); keep_mask = scores >= score_threshold
```

## 3. 接口规范

```python
cann_bench.bbox_decode_and_filter(Tensor anchors, Tensor deltas, Tensor scores, float score_threshold=0.05, float max_shape_h=1024.0, float max_shape_w=1024.0) -> (Tensor decoded_boxes, Tensor keep_mask)
```

### 输入参数

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `anchors` | `[N,4]` | float16 / bfloat16 / float32 | anchor boxes，格式 [x1,y1,x2,y2] |
| `deltas` | `[N,4]` | float16 / bfloat16 / float32 | bbox deltas，格式 [dx,dy,dw,dh] |
| `scores` | `[N]` | float16 / bfloat16 / float32 | 候选框分数 |
| `score_threshold` | 标量 | float | 保留候选框的分数阈值 |
| `max_shape_h` | 标量 | float | decode 后 box y 坐标裁剪上界 |
| `max_shape_w` | 标量 | float | decode 后 box x 坐标裁剪上界 |

### 输出

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `decoded_boxes` | `[N,4]` | 与主浮点输入相同 | decode 后 boxes |
| `keep_mask` | `[N]` | bool | score 过滤 mask |

### 数据类型

| 输入 dtype | 输出 dtype |
|---|---|
| float16 | float16 |
| bfloat16 | bfloat16 |
| float32 | float32 |

### 规则与约束

- 输入 shape 需满足 `proto.yaml` 中声明的维度关系。
- 所有整数属性需处于有效范围。
- 参考实现仅依赖 PyTorch，输出 dtype 与算子定义保持一致。

## 4. 精度要求

采用[生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)进行验证。

**误差指标**：MERE / MARE；bool 输出按精确匹配处理。

## 5. 标准 Golden 代码

```python
#!/usr/bin/python3
# coding=utf-8

import torch


def bbox_decode_and_filter(
    anchors: torch.Tensor,
    deltas: torch.Tensor,
    scores: torch.Tensor,
    score_threshold: float = 0.05,
    max_shape_h: float = 1024.0,
    max_shape_w: float = 1024.0,
):
    if anchors.dim() != 2 or deltas.dim() != 2 or anchors.shape[1] != 4 or deltas.shape[1] != 4:
        raise ValueError("anchors and deltas must be [N,4]")
    if anchors.shape != deltas.shape or scores.shape != (anchors.shape[0],):
        raise ValueError("anchors/deltas/scores shape mismatch")
    out_dtype = anchors.dtype
    a = anchors.float()
    d = deltas.float()
    x1 = torch.minimum(a[:, 0], a[:, 2])
    y1 = torch.minimum(a[:, 1], a[:, 3])
    x2 = torch.maximum(a[:, 0], a[:, 2])
    y2 = torch.maximum(a[:, 1], a[:, 3])
    widths = torch.clamp(x2 - x1, min=1.0)
    heights = torch.clamp(y2 - y1, min=1.0)
    ctr_x = x1 + 0.5 * widths
    ctr_y = y1 + 0.5 * heights
    dx, dy = d[:, 0], d[:, 1]
    dw = torch.clamp(d[:, 2], min=-4.0, max=4.0)
    dh = torch.clamp(d[:, 3], min=-4.0, max=4.0)
    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights
    decoded = torch.stack((
        pred_ctr_x - 0.5 * pred_w,
        pred_ctr_y - 0.5 * pred_h,
        pred_ctr_x + 0.5 * pred_w,
        pred_ctr_y + 0.5 * pred_h,
    ), dim=1)
    max_w = max(float(max_shape_w), 1.0)
    max_h = max(float(max_shape_h), 1.0)
    decoded[:, 0::2] = decoded[:, 0::2].clamp(0.0, max_w)
    decoded[:, 1::2] = decoded[:, 1::2].clamp(0.0, max_h)
    keep_mask = scores.float() >= float(score_threshold)
    return decoded.to(out_dtype), keep_mask
```
