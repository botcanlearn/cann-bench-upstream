/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

/**
 * @file sqrt.h
 * @brief ACLNN L0 API declaration
 */

#ifndef OP_API_INC_SQRT_H_
#define OP_API_INC_SQRT_H_

#include "opdev/op_executor.h"

namespace l0op {

const aclTensor* Sqrt(const aclTensor* x, aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_INC_SQRT_H_