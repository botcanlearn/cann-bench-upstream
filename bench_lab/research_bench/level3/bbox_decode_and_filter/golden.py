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
