#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

import pytest
import torch
import cann_bench


def test_add_float32():
    """Test Add operator with float32"""
    x = torch.randn(1024, dtype=torch.float32, device="npu")
    y = torch.randn(1024, dtype=torch.float32, device="npu")
    z = cann_bench.add(x, y)
    # Use CPU to compute expected (NPU builtin op may be affected by ASCEND_CUSTOM_OPP_PATH)
    expected = x.cpu() + y.cpu()
    assert torch.allclose(z.cpu(), expected, rtol=1e-5, atol=1e-5)


def test_add_float16():
    """Test Add operator with float16"""
    x = torch.randn(1024, dtype=torch.float16, device="npu")
    y = torch.randn(1024, dtype=torch.float16, device="npu")
    z = cann_bench.add(x, y)
    # Use CPU to compute expected
    expected = x.cpu() + y.cpu()
    assert torch.allclose(z.cpu(), expected, rtol=1e-3, atol=1e-3)


def test_add_int32():
    """Test Add operator with int32"""
    x = torch.randint(0, 100, (1024,), dtype=torch.int32, device="npu")
    y = torch.randint(0, 100, (1024,), dtype=torch.int32, device="npu")
    z = cann_bench.add(x, y)
    # Use CPU to compute expected
    expected = x.cpu() + y.cpu()
    assert torch.equal(z.cpu(), expected)


def test_add_via_torch_ops():
    """Test Add via torch.ops.cann_bench.add()"""
    x = torch.randn(1024, dtype=torch.float32, device="npu")
    y = torch.randn(1024, dtype=torch.float32, device="npu")
    z = torch.ops.cann_bench.add(x, y)
    # Use CPU to compute expected
    expected = x.cpu() + y.cpu()
    assert torch.allclose(z.cpu(), expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])