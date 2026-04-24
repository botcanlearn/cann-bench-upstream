/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#ifndef OP_API_INC_MISH_H_
#define OP_API_INC_MISH_H_

#include "opdev/op_executor.h"

namespace l0op {

const aclTensor* Mish(const aclTensor* x, aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_INC_MISH_H_
