# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------

# find ascend toolkit
if(UNIX)
  set(SYSTEM_PREFIX ${CMAKE_SYSTEM_PROCESSOR}-linux)
endif()

if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_DIR $ENV{ASCEND_HOME_PATH})
else()
    message(WARNING "Environment variable ASCEND_HOME_PATH is not set. Using default path.")
    if("$ENV{USER}" STREQUAL "root")
        message(STATUS "Running as root user, checking default root paths for Ascend toolkit.")
        if(EXISTS /usr/local/Ascend/ascend-toolkit/latest)
            set(ASCEND_DIR /usr/local/Ascend/ascend-toolkit/latest)
        elseif(EXISTS /usr/local/Ascend/latest)
            set(ASCEND_DIR /usr/local/Ascend/latest)
        else()
            message(FATAL_ERROR "Ascend toolkit not found in default root paths. Please set ASCEND_HOME_PATH.")
        endif()
    else()
        message(STATUS "Running as non-root user, checking default user paths for Ascend toolkit.")
        if(EXISTS $ENV{HOME}/Ascend/ascend-toolkit/latest)
            set(ASCEND_DIR $ENV{HOME}/Ascend/ascend-toolkit/latest)
        elseif(EXISTS $ENV{HOME}/Ascend/latest)
            set(ASCEND_DIR $ENV{HOME}/Ascend/latest)
        else()
            message(FATAL_ERROR "Ascend toolkit not found in default user paths. Please set ASCEND_HOME_PATH.")
        endif()
    endif()
endif()

message(STATUS "Using Ascend toolkit path: ${ASCEND_DIR}")
set(CMAKE_PREFIX_PATH ${ASCEND_DIR}/)
# NOTE: asc-config.cmake lives directly under <root>/lib64/cmake/ without a
# package-name subdirectory (e.g. ASC/).  A plain prefix like <root>/ won't
# match find_package(ASC)'s search rules, so we append the cmake config dir.
list(APPEND CMAKE_PREFIX_PATH ${ASCEND_DIR}/lib64/cmake)
set(BISHENG "${ASCEND_DIR}/${SYSTEM_PREFIX}/ccec_compiler/bin/bisheng" CACHE FILEPATH "Path to Bisheng compiler")
message(STATUS "Bisheng compiler path: ${BISHENG}")

# set ASCEND_INCLUDE_DIRS
set(ASCEND_INCLUDE_DIRS
    ${ASCEND_DIR}/include
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/include
    ${ASCEND_DIR}/compiler/tikcpp/include
    ${ASCEND_DIR}/compiler/ascendc/include/basic_api/impl
    ${ASCEND_DIR}/compiler/ascendc/include/basic_api/interface
    ${ASCEND_DIR}/compiler/ascendc/include/highlevel_api/impl
    ${ASCEND_DIR}/compiler/ascendc/include/highlevel_api/tiling
    ${ASCEND_DIR}/compiler/ascendc/impl/aicore/basic_api
)