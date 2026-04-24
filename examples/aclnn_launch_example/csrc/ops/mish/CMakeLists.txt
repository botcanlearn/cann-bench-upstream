# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# Mish算子自注册

set(MISH_HOST_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/mish_def.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/mish_infershape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/mish_tiling.cpp
)

set(MISH_API_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_api/aclnn_mish.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_api/mish.cpp
)

register_aclnn_op(
    Mish
    "${MISH_HOST_SRCS}"
    "${MISH_API_SRCS}"
    mish/op_kernel
    mish_kernel.cpp
    op_kernel
    op_api
)

set(MISH_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/mish_plugin.cpp
)
register_aclnn_plugin("${MISH_PLUGIN_SRCS}" op_api)
