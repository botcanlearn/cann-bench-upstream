#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

import pytest
import torch
import cann_bench


def test_sqrt_float32():
    """Test Sqrt operator with float32"""
    x = torch.rand(1024, dtype=torch.float32, device="npu")
    y = cann_bench.sqrt(x)
    # Use CPU to compute expected (NPU builtin op may be affected by ASCEND_CUSTOM_OPP_PATH)
    expected = torch.sqrt(x.cpu())
    assert torch.allclose(y.cpu(), expected, rtol=1e-5, atol=1e-5)


def test_sqrt_float16():
    """Test Sqrt operator with float16"""
    x = torch.rand(1024, dtype=torch.float16, device="npu")
    y = cann_bench.sqrt(x)
    # Use CPU to compute expected
    expected = torch.sqrt(x.cpu())
    assert torch.allclose(y.cpu(), expected, rtol=1e-3, atol=1e-3)


def test_sqrt_via_torch_ops():
    """Test Sqrt via torch.ops.cann_bench.sqrt()"""
    x = torch.rand(1024, dtype=torch.float32, device="npu")
    y = torch.ops.cann_bench.sqrt(x)
    # Use CPU to compute expected
    expected = torch.sqrt(x.cpu())
    assert torch.allclose(y.cpu(), expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])