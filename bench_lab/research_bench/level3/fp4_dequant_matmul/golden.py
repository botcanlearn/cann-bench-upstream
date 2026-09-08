#!/usr/bin/python3
# coding=utf-8

import torch


_FP4_E2M1_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _unpack_nibbles(packed: torch.Tensor, n_values: int) -> torch.Tensor:
    values = packed.to(torch.uint8).reshape(-1)
    low = values & 0x0F
    high = (values >> 4) & 0x0F
    codes = torch.stack((low, high), dim=1).reshape(-1)
    return codes[:n_values].to(torch.long)


def fp4_dequant_matmul(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    out_features: int,
    in_features: int,
    block_size: int = 16,
) -> torch.Tensor:
    if x.shape[-1] != in_features:
        raise ValueError("x.shape[-1] must equal in_features")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if scales.dim() != 2 or scales.shape[0] != out_features:
        raise ValueError("scales must be [out_features, ceil(in_features / block_size)]")

    total = out_features * in_features
    codes = _unpack_nibbles(w_packed, total)
    table = _FP4_E2M1_TABLE.to(device=x.device)
    weight = table.index_select(0, codes.to(x.device)).reshape(out_features, in_features)

    scale_blocks = scales.float()
    expanded_scales = scale_blocks.repeat_interleave(block_size, dim=1)[:, :in_features]
    weight = weight * expanded_scales.to(weight.device)

    out = torch.matmul(x.float(), weight.t())
    return out.to(x.dtype)
