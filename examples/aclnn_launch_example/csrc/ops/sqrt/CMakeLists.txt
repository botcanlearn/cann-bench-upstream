# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# Sqrt算子自注册

# Host源文件
set(SQRT_HOST_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/sqrt_def.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/sqrt_infershape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/sqrt_tiling.cpp
)

# API源文件
set(SQRT_API_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_api/aclnn_sqrt.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_api/sqrt.cpp
)

# 注册到全局列表
register_aclnn_op(
    Sqrt
    "${SQRT_HOST_SRCS}"
    "${SQRT_API_SRCS}"
    sqrt/op_kernel
    sqrt.cpp
    op_kernel
    op_api
)

# 注册Python插件(可选，用于Python扩展构建)
set(SQRT_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/sqrt_plugin.cpp
)
register_aclnn_plugin("${SQRT_PLUGIN_SRCS}" op_api)