/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 */

/**
 * @file add.h
 * @brief ACLNN L0 API declaration
 */

#ifndef OP_API_INC_ADD_H_
#define OP_API_INC_ADD_H_

#include "opdev/op_executor.h"

namespace l0op {

const aclTensor* Add(const aclTensor* x1, const aclTensor* x2, aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_INC_ADD_H_