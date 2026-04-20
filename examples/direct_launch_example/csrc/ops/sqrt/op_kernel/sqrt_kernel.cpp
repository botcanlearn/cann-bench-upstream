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
 * \file sqrt_host.cpp
 * \brief Sqrt host code - kernel launch and tiling (compiled with bisheng + -xasc)
 */

#include <tuple>
#include <algorithm>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

constexpr static int64_t PIPELINE_DEPTH = 2;

template <typename T>
class KernelSqrt {
public:
    __aicore__ inline KernelSqrt() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, int64_t totalLength, int64_t blockLength, uint32_t tileSize)
    {
        xGm_.SetGlobalBuffer((__gm__ T *)x + blockLength * AscendC::GetBlockIdx());
        zGm_.SetGlobalBuffer((__gm__ T *)z + blockLength * AscendC::GetBlockIdx());
        pipe_.InitBuffer(inQueueX_, PIPELINE_DEPTH, tileSize);
        pipe_.InitBuffer(outQueueZ_, PIPELINE_DEPTH, tileSize);
        int64_t currentBlockLength = totalLength - AscendC::GetBlockIdx() * blockLength;
        if (currentBlockLength > blockLength) currentBlockLength = blockLength;
        elementNumPerTile_ = tileSize / sizeof(T);
        tileNum_ = currentBlockLength / elementNumPerTile_;
        tailTileElementNum_ = currentBlockLength - tileNum_ * elementNumPerTile_;
    }

    __aicore__ inline void Process()
    {
        for (int64_t i = 0; i < tileNum_; ++i) {
            CopyIn(i * elementNumPerTile_, elementNumPerTile_);
            Compute(elementNumPerTile_);
            CopyOut(i * elementNumPerTile_, elementNumPerTile_);
        }
        if (tailTileElementNum_ > 0) {
            CopyIn(tileNum_ * elementNumPerTile_, tailTileElementNum_);
            Compute(tailTileElementNum_);
            CopyOut(tileNum_ * elementNumPerTile_, tailTileElementNum_);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t offset, int64_t count)
    {
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        auto xLocal = inQueueX_.AllocTensor<T>();
        AscendC::DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int64_t count)
    {
        auto xLocal = inQueueX_.DeQue<T>();
        auto zLocal = outQueueZ_.AllocTensor<T>();
        AscendC::Sqrt(zLocal, xLocal, count);
        outQueueZ_.EnQue(zLocal);
        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t offset, int64_t count)
    {
        auto zLocal = outQueueZ_.DeQue<T>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(zGm_[offset], zLocal, copyParams);
        outQueueZ_.FreeTensor(zLocal);
    }

    AscendC::TPipe pipe_;
    AscendC::GlobalTensor<T> xGm_, zGm_;
    AscendC::TQue<AscendC::TPosition::VECIN, PIPELINE_DEPTH> inQueueX_;
    AscendC::TQue<AscendC::TPosition::VECOUT, PIPELINE_DEPTH> outQueueZ_;
    int64_t elementNumPerTile_ = 0, tileNum_ = 0, tailTileElementNum_ = 0;
};

template <typename T>
__global__ __aicore__ __vector__ void sqrt_kernel(GM_ADDR x, GM_ADDR z, int64_t totalLength, int64_t blockLength, uint32_t tileSize)
{
    KernelSqrt<T> op;
    op.Init(x, z, totalLength, blockLength, tileSize);
    op.Process();
}

std::tuple<int64_t, int64_t, int64_t> calc_sqrt_tiling_params(int64_t totalLength)
{
    constexpr static int64_t MIN_ELEMS_PER_CORE = 1024;
    constexpr static int64_t BUFFER_NUM = 2;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) coreNum = 1;
    int64_t numBlocks = std::min(coreNum, (totalLength + MIN_ELEMS_PER_CORE - 1) / MIN_ELEMS_PER_CORE);
    numBlocks = std::max(numBlocks, static_cast<int64_t>(1));
    int64_t blockLength = (totalLength + numBlocks - 1) / numBlocks;
    int64_t tileSize = ubSize / PIPELINE_DEPTH / BUFFER_NUM;
    return std::make_tuple(numBlocks, blockLength, tileSize);
}

extern "C" {

void launch_sqrt_kernel_float(GM_ADDR x, GM_ADDR z, int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream)
{
    sqrt_kernel<float><<<numBlocks, nullptr, stream>>>(x, z, totalLength, blockLength, tileSize);
}

}