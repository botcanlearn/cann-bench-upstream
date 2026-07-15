/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_plugin.cpp
 * \brief Add plugin layer - torch bindings (compiled with g++)
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include "../../_common/aclnn_common.h"

namespace cann_bench {

static torch::Tensor add_meta(const torch::Tensor& x1, const torch::Tensor& x2) {
    TORCH_CHECK(x1.sizes() == x2.sizes(),
                "add: shapes must match, got ", x1.sizes(), " vs ", x2.sizes());
    TORCH_CHECK(x1.scalar_type() == x2.scalar_type(),
                "add: dtypes must match, got ", x1.scalar_type(), " vs ", x2.scalar_type());
    return torch::empty_like(x1);
}

static torch::Tensor add_npu(const torch::Tensor& x1, const torch::Tensor& x2) {
    auto x1_c = x1.contiguous();
    auto x2_c = x2.contiguous();
    auto y = torch::empty_like(x1_c);
    ACLNN_CMD(aclnnAdd, x1_c, x2_c, y);
    return y;
}

TORCH_LIBRARY_FRAGMENT(cann_bench, m) {
    m.def("add(Tensor x1, Tensor x2) -> Tensor");
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m) {
    m.impl("add", add_meta);
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m) {
    m.impl("add", add_npu);
}

}  // namespace cann_bench
