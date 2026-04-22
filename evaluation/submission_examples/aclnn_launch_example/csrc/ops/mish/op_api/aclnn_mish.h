/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 */

#ifndef ACLNN_MISH_H_
#define ACLNN_MISH_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnMishGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

ACLNN_API aclnnStatus aclnnMish(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_MISH_H_
