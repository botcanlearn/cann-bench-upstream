/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */
#pragma once

#include <tuple>
#include <cstdint>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

std::tuple<int64_t, int64_t, int64_t> calc_device_memcpy_tiling_params(int64_t totalBytes);

extern "C" {
void launch_device_memcpy_kernel(GM_ADDR src, GM_ADDR dst, int64_t totalBytes,
                                  int64_t blocks, int64_t blkBytes,
                                  uint32_t tileSz, void *stream);
}

#define DEVICE_MEMCPY_KERNEL_LAUNCH(s, d, n, b, l, t, st) \
    launch_device_memcpy_kernel(s, d, n, b, l, t, st)
