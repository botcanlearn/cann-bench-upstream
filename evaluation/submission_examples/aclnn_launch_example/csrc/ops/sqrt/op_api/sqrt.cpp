/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 */

/**
 * @file sqrt.cpp
 * @brief ACLNN L0 API implementation
 */

#include "sqrt.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Sqrt);

static const std::initializer_list<op::DataType> DTYPE_SUPPORT = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16
};

static bool IsAiCoreSupport(const aclTensor* x)
{
    return CheckType(x->GetDataType(), DTYPE_SUPPORT);
}

static const aclTensor* SqrtAiCore(const aclTensor* x, const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(SqrtAiCore, x, out);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Sqrt, OP_INPUT(x), OP_OUTPUT(out));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "SqrtAiCore failed."), return nullptr);
    return out;
}

const aclTensor* Sqrt(const aclTensor* x, aclOpExecutor* executor)
{
    const aclTensor* out = nullptr;

    if (!IsAiCoreSupport(x)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Sqrt not supported: dtype=%d", static_cast<int>(x->GetDataType()));
        return nullptr;
    }

    out = executor->AllocTensor(x->GetViewShape(), x->GetDataType());
    return SqrtAiCore(x, out, executor);
}

} // namespace l0op