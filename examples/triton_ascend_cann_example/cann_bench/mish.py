"""Triton-Ascend implementation of the CANN Bench Mish interface."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cann import libdevice


_BLOCK_SIZE = 4096


@triton.jit
def _mish_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    value = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # max(x, 0) + log1p(exp(-abs(x))) is stable at both infinities.
    positive = tl.where(value > 0.0, value, 0.0)
    softplus = positive + libdevice.log1p(tl.exp(-tl.abs(value)))
    result = value * libdevice.tanh(softplus)
    tl.store(output_ptr + offsets, result, mask=mask)


def mish(x: torch.Tensor) -> torch.Tensor:
    """Return the element-wise Mish activation of an NPU tensor."""
    if not x.is_contiguous():
        x = x.contiguous()

    output = torch.empty_like(x)
    n_elements = x.numel()
    if n_elements == 0:
        return output

    grid = (triton.cdiv(n_elements, _BLOCK_SIZE),)
    _mish_kernel[grid](x, output, n_elements, BLOCK_SIZE=_BLOCK_SIZE)
    return output
