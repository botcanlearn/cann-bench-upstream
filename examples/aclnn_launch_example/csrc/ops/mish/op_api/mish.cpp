/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 */

#include "mish.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Mish);

static const std::initializer_list<op::DataType> DTYPE_SUPPORT = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16
};

static bool IsAiCoreSupport(const aclTensor* x)
{
    return CheckType(x->GetDataType(), DTYPE_SUPPORT);
}

static const aclTensor* MishAiCore(const aclTensor* x,
                                    const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(MishAiCore, x, out);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Mish, OP_INPUT(x), OP_OUTPUT(out));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "MishAiCore failed."), return nullptr);
    return out;
}

const aclTensor* Mish(const aclTensor* x, aclOpExecutor* executor)
{
    if (!IsAiCoreSupport(x)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Mish not supported: dtype=%d", static_cast<int>(x->GetDataType()));
        return nullptr;
    }
    const aclTensor* out = executor->AllocTensor(x->GetViewShape(), x->GetDataType());
    return MishAiCore(x, out, executor);
}

} // namespace l0op
