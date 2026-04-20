# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# Add算子自注册

# Host源文件
set(ADD_HOST_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/add_def.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/add_infershape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/add_tiling.cpp
)

# API源文件
set(ADD_API_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_api/aclnn_add.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_api/add.cpp
)

# 注册到全局列表
register_aclnn_op(
    Add
    "${ADD_HOST_SRCS}"
    "${ADD_API_SRCS}"
    add/op_kernel
    add.cpp
    op_kernel
    op_api
)

# 注册Python插件(可选，用于Python扩展构建)
set(ADD_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/add_plugin.cpp
)
register_aclnn_plugin("${ADD_PLUGIN_SRCS}" op_api)