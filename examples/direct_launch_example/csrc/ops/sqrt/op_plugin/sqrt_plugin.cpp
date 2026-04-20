/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sqrt_api.cpp
 * \brief Sqrt API layer - torch bindings (compiled with g++)
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/sqrt_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("sqrt(Tensor x) -> Tensor");
}

torch::Tensor sqrt_meta(const torch::Tensor &x)
{
    return torch::empty_like(x);
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("sqrt", sqrt_meta);
}

torch::Tensor sqrt_npu(const torch::Tensor &x)
{
    const c10::OptionalDeviceGuard guard(x.device());
    auto z = sqrt_meta(x);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    int64_t totalLength = x.numel();
    int64_t numBlocks, blockLength, tileSize;
    std::tie(numBlocks, blockLength, tileSize) = calc_sqrt_tiling_params(totalLength);
    auto x_ptr = (GM_ADDR)x.data_ptr();
    auto z_ptr = (GM_ADDR)z.data_ptr();

    auto acl_call = [=]() -> int {
        if (x.scalar_type() == torch::kFloat32) {
            SQRT_KERNEL_LAUNCH_FLOAT(x_ptr, z_ptr, totalLength, numBlocks, blockLength, tileSize, stream);
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("Sqrt", acl_call);
    return z;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("sqrt", sqrt_npu);
}

} // namespace cann_bench