#!/usr/bin/env python3

import torch
import torch_npu
import cann_bench
import pytest

def test_add_works():
    assert hasattr(cann_bench, "add")

SHAPES = [(1,), (1024,), (10, 10), (256, 512), (4, 3, 64, 64)]
DTYPES = [torch.float32, torch.float16, torch.int32]

@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not found")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_add(shape, dtype):
    if dtype == torch.int32:
        a = torch.randint(-100, 100, shape, dtype=dtype).npu()
        b = torch.randint(-100, 100, shape, dtype=dtype).npu()
        expected = a + b
        result = cann_bench.add(a, b).cpu()
        assert torch.equal(result, expected.cpu())
    else:
        a = torch.randn(shape, dtype=dtype).npu()
        b = torch.randn(shape, dtype=dtype).npu()
        expected = a + b
        result = cann_bench.add(a, b).cpu()
        assert torch.allclose(result, expected.cpu(), rtol=1e-4)