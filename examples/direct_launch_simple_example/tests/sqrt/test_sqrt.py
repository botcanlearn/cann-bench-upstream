#!/usr/bin/env python3

import torch
import torch_npu
import cann_bench
import pytest

def test_sqrt_works():
    assert hasattr(cann_bench, "sqrt")

SHAPES = [(1,), (1024,), (10, 10), (256, 512), (4, 3, 64, 64)]

@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not found")
@pytest.mark.parametrize("shape", SHAPES)
def test_sqrt(shape):
    a = torch.abs(torch.randn(shape)).npu()
    expected = torch.sqrt(a)
    result = cann_bench.sqrt(a).cpu()
    assert torch.allclose(result, expected.cpu(), rtol=1e-4)