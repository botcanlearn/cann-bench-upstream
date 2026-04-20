/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 */

#include <torch/extension.h>
#include "ops/add/op_kernel/add_launch.h"
#include "ops/sqrt/op_kernel/sqrt_launch.h"

// Forward declarations from op_api files
torch::Tensor add_npu(const torch::Tensor &x, const torch::Tensor &y);
torch::Tensor sqrt_npu(const torch::Tensor &x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add_npu, "Add two tensors on NPU");
    m.def("sqrt", &sqrt_npu, "Sqrt tensor on NPU");
}