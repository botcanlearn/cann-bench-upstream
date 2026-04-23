#!/usr/bin/env python3
"""
ROI Pooling 用例验证脚本
仅使用 CPU torch 和手写实现进行验证
"""

import torch
import yaml
import sys

# dtype 映射
dtype_map = {
    'float16': torch.float16,
    'float32': torch.float32,
}


def roi_pooling_manual(
    x: torch.Tensor, rois: torch.Tensor, pooled_h: int, pooled_w: int,
    spatial_scale: float = 1.0
) -> torch.Tensor:
    """
    手写实现：对输入特征图按ROI进行最大池化
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


def main():
    # 加载 cases
    with open('cases.yaml') as f:
        cases = yaml.safe_load(f)['cases']

    print(f"验证 ROI Pooling 共 {len(cases)} 个用例")
    print("-" * 50)

    errors = []
    for case in cases:
        case_id = case['case_id']
        x_shape, rois_shape = case['input_shape']
        dtype_x = case['dtype'][0]
        attrs = case['attrs']

        torch_dtype = dtype_map.get(dtype_x, torch.float32)

        # 生成数据（CPU）
        x = torch.randn(x_shape, dtype=torch_dtype, device='cpu')

        # 生成有效 ROI
        num_rois = rois_shape[0]
        batch_size = x_shape[0]
        H, W = x_shape[2], x_shape[3]
        scale = attrs['spatial_scale']

        # ROI 原图坐标，映射后应在特征图范围内
        rois = torch.zeros(rois_shape, dtype=torch_dtype, device='cpu')
        for i in range(num_rois):
            rois[i, 0] = i % batch_size
            # 确保 ROI 足够大，映射后覆盖特征图的 20%-80% 区域
            roi_h_scaled = H * 0.6  # 映射后的 ROI 高度
            roi_w_scaled = W * 0.6  # 映射后的 ROI 宽度
            rois[i, 1] = (H * 0.2) / scale  # x1
            rois[i, 2] = (W * 0.2) / scale  # y1
            rois[i, 3] = (H * 0.2 + roi_h_scaled) / scale  # x2
            rois[i, 4] = (W * 0.2 + roi_w_scaled) / scale  # y2

        try:
            y = roi_pooling_manual(x, rois, attrs['pooled_h'], attrs['pooled_w'], attrs['spatial_scale'])

            expected_shape = [num_rois, x_shape[1], attrs['pooled_h'], attrs['pooled_w']]
            if list(y.shape) == expected_shape:
                # 检查输出是否有效（不全为 0）
                non_zero_ratio = (y != 0).float().mean().item()
                print(f"case {case_id}: ✓ shape={list(y.shape)}, dtype={y.dtype}, non_zero_ratio={non_zero_ratio:.2%}")
            else:
                errors.append(f"case {case_id}: shape mismatch, got {list(y.shape)}, expected {expected_shape}")
        except Exception as e:
            errors.append(f"case {case_id}: {e}")

    print("-" * 50)
    if errors:
        print("=== Errors ===")
        for e in errors:
            print(e)
        sys.exit(1)
    else:
        print(f"全部 {len(cases)} 个用例验证通过!")
        sys.exit(0)


if __name__ == '__main__':
    main()