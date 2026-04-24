/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace cann_bench {

static torch::Tensor mish_meta(const torch::Tensor& x) {
    return torch::empty_like(x);
}

static torch::Tensor mish_npu(const torch::Tensor& x) {
    auto y = mish_meta(x);
    at_npu::native::OpCommand cmd;
    cmd.Name("Mish")
       .Input(x)
       .Output(y)
       .Run();
    return y;
}

TORCH_LIBRARY_FRAGMENT(cann_bench, m) {
    m.def("mish(Tensor x) -> Tensor");
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m) {
    m.impl("mish", mish_meta);
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m) {
    m.impl("mish", mish_npu);
}

} // namespace cann_bench
