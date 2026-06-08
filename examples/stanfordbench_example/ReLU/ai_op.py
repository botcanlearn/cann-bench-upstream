# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You can not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
StanfordBench AI 算子示例：ReLU

这是一个 Triton 实现的 ReLU 激活函数，用于演示如何测试 StanfordBench 格式的 AI 算子。

测试方式:
    ./scripts/run_evaluation.sh --bench-name stanford --task-dir bench_lab/stanford_bench/KernelBench/KernelBench --operator ReLU --source-dir examples/stanfordbench_example/ReLU
"""

import torch
import triton
import triton.language as tl


@triton.jit
def akg_agents_relu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    CORE_NUM: tl.constexpr,
):
    """
    ReLU激活函数内核。
    每个程序（核心）交错处理多个数据块，实现负载均衡。
    Args:
        input_ptr: 输入张量指针。
        output_ptr: 输出张量指针。
        n_elements: 输入张量的总元素数。
        BLOCK_SIZE: 每个数据块处理的元素数。
        CORE_NUM: 启动的核心数（VEC核心数）。
    """
    pid = tl.program_id(0)

    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)

    for block_idx in range(pid, num_blocks, CORE_NUM):
        block_start = block_idx * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        data = tl.load(input_ptr + offsets, mask=mask, other=0.0)

        result = tl.maximum(data, 0.0)

        tl.store(output_ptr + offsets, result, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        try:
            import torch_npu
            self.VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
        except:
            self.VEC_CORE_NUM = 40

        self.BLOCK_SIZE = 4096

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入张量应用ReLU激活。
        Args:
            x (torch.Tensor): 输入张量，任意形状。
        Returns:
            torch.Tensor: 输出张量，与输入形状相同。
        """
        if not x.is_contiguous():
            x = x.contiguous()

        output = torch.empty_like(x)

        n_elements = x.numel()

        grid = (self.VEC_CORE_NUM,)

        akg_agents_relu_kernel[grid](
            x,
            output,
            n_elements,
            BLOCK_SIZE=self.BLOCK_SIZE,
            CORE_NUM=self.VEC_CORE_NUM,
        )

        return output