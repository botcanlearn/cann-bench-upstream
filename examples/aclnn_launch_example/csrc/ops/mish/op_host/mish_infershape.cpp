/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4Mish(gert::InferShapeContext* context)
{
    const gert::Shape* input_shape = context->GetInputShape(0);
    if (input_shape == nullptr) return ge::GRAPH_FAILED;
    gert::Shape* output_shape = context->GetOutputShape(0);
    if (output_shape == nullptr) return ge::GRAPH_FAILED;
    *output_shape = *input_shape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Mish).InferShape(InferShape4Mish);

} // namespace ops
