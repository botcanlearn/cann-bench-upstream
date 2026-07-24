"""Triton-Ascend implementation of the CANN Bench Sigmoid interface."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_BLOCK_SIZE = 4096


@triton.jit
def _sigmoid_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    value = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(output_ptr + offsets, tl.sigmoid(value), mask=mask)


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Return the element-wise sigmoid of an NPU tensor."""
    if not x.is_contiguous():
        x = x.contiguous()

    output = torch.empty_like(x)
    n_elements = x.numel()
    if n_elements == 0:
        return output

    grid = (triton.cdiv(n_elements, _BLOCK_SIZE),)
    _sigmoid_kernel[grid](x, output, n_elements, BLOCK_SIZE=_BLOCK_SIZE)
    return output
