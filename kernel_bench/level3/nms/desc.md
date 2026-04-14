# NMS 算子 API 描述

## 1. 算子简介

对候选框执行非极大值抑制 (Non-Maximum Suppression)，根据置信度分数和 IoU 阈值过滤重叠的候选框。

**主要应用场景**：
- 目标检测模型（如 YOLO、Faster R-CNN）的后处理去重
- 实例分割中候选区域的筛选
- 人脸检测等需要消除重叠检测框的场景

**算子特征**：
- 难度等级：L3（SortSelect）
- 双输入单输出，根据置信度排序并基于 IoU 阈值迭代过滤重叠框，输出保留框的索引

## 2. 算子定义

### 数学公式

$$
\text{keep\_indices} = \text{NMS}(\text{boxes}, \text{scores}, \text{iou\_threshold})
$$

### 处理流程

1. 按 scores 从高到低对候选框排序
2. 选取得分最高的框加入保留列表
3. 计算该框与剩余所有框的 IoU（交并比）
4. 移除 IoU 大于 `iou_threshold` 的重叠框
5. 重复步骤 2-4，直至所有框被处理

其中 IoU 定义为：

$$
\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

## 3. 接口规范

### 算子原型

```python
ascend_bench.nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor keep_indices
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| boxes | Tensor | 必选 | 输入候选框，格式为 [x1, y1, x2, y2]，shape 为 [N, 4] |
| scores | Tensor | 必选 | 每个候选框的置信度分数，shape 为 [N] |
| iou_threshold | float | 必选 | IoU 阈值，用于过滤重叠框 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| keep_indices | [M] | int64 | NMS 后保留的框索引，M <= N |

### 数据类型

| 输入 (boxes) dtype | 输入 (scores) dtype | 输出 dtype |
|-------------------|-------------------|-----------|
| float32 | float32 | int64 |

### 规则与约束

- `boxes` 的形状必须为 [N, 4]，每行为 [x1, y1, x2, y2] 格式
- `scores` 的形状必须为 [N]，且 N 与 boxes 的第一维一致
- `iou_threshold` 取值范围为 (0, 1)，值越小过滤越严格
- 输出 `keep_indices` 为 1D int64 张量，长度 M 取决于过滤后保留的框数
- 输出索引按置信度从高到低排序

## 4. 精度要求

计算结果与 PyTorch Golden 实现逐元素对比，需满足以下误差阈值：

| 数据类型 | 验证方式 | rtol | atol |
|---------|---------|------|------|
| float32 | 相对误差 | 1e-4 | 1e-4 |
| int64 (输出) | 完全相等 | — | — |

**对比公式**：

$$
|output - golden| \leq atol + rtol \times |golden|
$$

## 5. 标准 Golden 代码

```python
import torch

"""
NMS 算子 Torch Golden 参考实现

对候选框执行非极大值抑制 (Non-Maximum Suppression)
公式：keep_indices = nms(boxes, scores, iou_threshold)
"""
def nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float
) -> torch.Tensor:
    """
    对候选框执行非极大值抑制

    公式：keep_indices = nms(boxes, scores, iou_threshold)

    Args:
        boxes: 输入候选框，格式为 [x1, y1, x2, y2]，shape 为 [N, 4]
        scores: 每个候选框的置信度分数，shape 为 [N]
        iou_threshold: IoU 阈值，用于过滤重叠框

    Returns:
        keep_indices: NMS 后保留的框索引，shape 为 [M]
    """

    # 确保输入格式正确
    assert boxes.dim() == 2 and boxes.shape[1] == 4, "boxes shape must be [N, 4]"
    assert scores.dim() == 1 and scores.shape[0] == boxes.shape[0], "scores shape must be [N]"

    # 纯 PyTorch 实现 NMS，避免 torchvision ABI 兼容问题
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    _, order = scores.sort(descending=True)

    keep = []
    while order.numel() > 0:
        if order.numel() == 0:
            break
        i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            order = order.new_empty(0)
            break

        xx1 = boxes[order[1:], 0].clamp(min=boxes[i, 0])
        yy1 = boxes[order[1:], 1].clamp(min=boxes[i, 1])
        xx2 = boxes[order[1:], 2].clamp(max=boxes[i, 2])
        yy2 = boxes[order[1:], 3].clamp(max=boxes[i, 3])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = (iou <= iou_threshold).nonzero(as_tuple=False).squeeze(1)

        order = order[inds + 1]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import ascend_bench

boxes = torch.rand(1000, 4, dtype=torch.float32, device="npu") * 100
scores = torch.rand(1000, dtype=torch.float32, device="npu")
keep = ascend_bench.nms(boxes, scores, iou_threshold=0.5)

# 低 IoU 阈值（更严格的过滤）
keep = ascend_bench.nms(boxes, scores, iou_threshold=0.3)
```

### 性能基线参考

基于 cases.yaml 中 20 个测试用例，所有用例的 baseline_perf_us 均未测量（None）。测试用例覆盖了 50~2000 个候选框、iou_threshold 从 0.1 到 0.9 的不同阈值、float16 和 float32 数据类型、对齐与质数数量框、零值和特殊值（inf/-inf）范围等场景。

### 相关算子

- **EmbeddingHashLookupOrInsert**：同为 L3 级别的索引相关算子
- **MoeReRouting**：同为 L3 级别的数据选择/重排算子
- **MoeGatingTopKSoftmax**：同为选择类算子，执行 TopK 选择操作
