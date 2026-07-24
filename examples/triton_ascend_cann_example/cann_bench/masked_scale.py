"""Triton-Ascend implementation of the CANN Bench MaskedScale interface."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_BLOCK_SIZE = 4096


@triton.jit
def _masked_scale_kernel(
    x_ptr,
    mask_ptr,
    output_ptr,
    n_elements,
    scale,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < n_elements
    value = tl.load(x_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    mask_value = tl.load(mask_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    tl.store(output_ptr + offsets, value * mask_value * scale, mask=valid)


def masked_scale(
    x: torch.Tensor,
    mask: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Return ``x * mask * scale`` with the result stored in ``x.dtype``."""
    if x.shape != mask.shape:
        raise ValueError("x and mask must have the same shape")
    if not x.is_contiguous():
        x = x.contiguous()
    if not mask.is_contiguous():
        mask = mask.contiguous()

    output = torch.empty_like(x)
    n_elements = x.numel()
    if n_elements == 0:
        return output

    grid = (triton.cdiv(n_elements, _BLOCK_SIZE),)
    _masked_scale_kernel[grid](
        x,
        mask,
        output,
        n_elements,
        scale,
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    return output
