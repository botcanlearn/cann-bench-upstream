// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#ifndef ADD_H
#define ADD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "add_tiling_data.h"
#include "add_tiling_key.h"

namespace NsAdd {

using namespace AscendC;

template <typename T>
struct CalcTypeTraits {
    using type = T;
};

template <>
struct CalcTypeTraits<bfloat16_t> {
    using type = float;
};

template <typename T, int BUFFER_MODE>
class Add {
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;
    using CalcT = typename CalcTypeTraits<T>::type;
    static constexpr bool NEED_CAST = !std::is_same<T, CalcT>::value;

public:
    __aicore__ inline Add(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const AddTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueZ;
    TBuf<TPosition::VECCALC> tempBufX;
    TBuf<TPosition::VECCALC> tempBufY;
    TBuf<TPosition::VECCALC> tempBufZ;
    GlobalTensor<T> inputGMX, inputGMY, outputGMZ;
    int64_t blockLength_ = 0, ubLength_ = 0;
};

template <typename T, int BUFFER_MODE>
__aicore__ inline void Add<T, BUFFER_MODE>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const AddTilingData* tilingData)
{
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    ubLength_ = tilingData->ubFactor;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + tilingData->blockFactor * GetBlockIdx(), blockLength_);
    inputGMY.SetGlobalBuffer((__gm__ T*)y + tilingData->blockFactor * GetBlockIdx(), blockLength_);
    outputGMZ.SetGlobalBuffer((__gm__ T*)z + tilingData->blockFactor * GetBlockIdx(), blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(inputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueZ, BUFFER_NUM, ubLength_ * sizeof(T));
    if constexpr (NEED_CAST) {
        pipe.InitBuffer(tempBufX, ubLength_ * sizeof(CalcT));
        pipe.InitBuffer(tempBufY, ubLength_ * sizeof(CalcT));
        pipe.InitBuffer(tempBufZ, ubLength_ * sizeof(CalcT));
    }
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Add<T, BUFFER_MODE>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.template AllocTensor<T>();
    LocalTensor<T> yLocal = inputQueueY.template AllocTensor<T>();
    DataCopyParams copyParams{1, static_cast<uint16_t>(currentNum * sizeof(T)), 0, 0};
    DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, {false, 0, 0, 0});
    DataCopyPad(yLocal, inputGMY[progress * ubLength_], copyParams, {false, 0, 0, 0});
    inputQueueX.EnQue(xLocal);
    inputQueueY.EnQue(yLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Add<T, BUFFER_MODE>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> zLocal = outputQueueZ.template DeQue<T>();
    DataCopyParams copyParams{1, static_cast<uint16_t>(currentNum * sizeof(T)), 0, 0};
    DataCopyPad(outputGMZ[progress * ubLength_], zLocal, copyParams);
    outputQueueZ.FreeTensor(zLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Add<T, BUFFER_MODE>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.template DeQue<T>();
    LocalTensor<T> yLocal = inputQueueY.template DeQue<T>();
    LocalTensor<T> zLocal = outputQueueZ.template AllocTensor<T>();

    if constexpr (NEED_CAST) {
        LocalTensor<CalcT> xCalc = tempBufX.Get<CalcT>();
        LocalTensor<CalcT> yCalc = tempBufY.Get<CalcT>();
        LocalTensor<CalcT> zCalc = tempBufZ.Get<CalcT>();
        Cast(xCalc, xLocal, RoundMode::CAST_NONE, currentNum);
        Cast(yCalc, yLocal, RoundMode::CAST_NONE, currentNum);
        AscendC::Add(zCalc, xCalc, yCalc, currentNum);
        Cast(zLocal, zCalc, RoundMode::CAST_RINT, currentNum);
    } else {
        AscendC::Add(zLocal, xLocal, yLocal, currentNum);
    }

    outputQueueZ.template EnQue<T>(zLocal);
    inputQueueX.FreeTensor(xLocal);
    inputQueueY.FreeTensor(yLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Add<T, BUFFER_MODE>::Process()
{
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsAdd
#endif // ADD_H