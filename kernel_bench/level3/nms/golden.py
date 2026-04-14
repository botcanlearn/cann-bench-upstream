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
