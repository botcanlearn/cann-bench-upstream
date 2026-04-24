/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */
#include "sqrt.h"

template <typename D_T_X, int BUFFER_MODE>
__global__ __aicore__ void sqrt(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SqrtTilingData);
    GET_TILING_DATA_WITH_STRUCT(SqrtTilingData, tilingData, tiling);
    NsSqrt::Sqrt<D_T_X, BUFFER_MODE> op;
    op.Init(x, y, &tilingData);
    op.Process();
}