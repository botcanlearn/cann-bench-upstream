/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 */
#include "mish.h"

template <typename D_T_X, int BUFFER_MODE>
__global__ __aicore__ void mish(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MishTilingData);
    GET_TILING_DATA_WITH_STRUCT(MishTilingData, tilingData, tiling);
    NsMish::Mish<D_T_X, BUFFER_MODE> op;
    op.Init(x, y, &tilingData);
    op.Process();
}
