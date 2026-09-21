#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import torch

"""
DynamicMxQuant 算子 Torch Golden 参考实现

MX (microscaling) 动态量化：在 axis 维上按 blocksize 分块，每块计算 E8M0 共享尺度
mxscale，块内元素除以 mxscale 后按 round_mode 舍入到 dst_type (FP8)。

公式 (标准 OCP MX 算法):
    shared_exp = floor(log2(max_i(|V_i|))) - emax
    mxscale    = 2^shared_exp
    P_i        = cast_to_dst_type(V_i / mxscale, round_mode)

mxscale 为逐块尺度张量：rank 与 x 一致，量化轴维 = ceil(x.shape[axis]/blocksize)，
第 b 块的尺度直接存于量化轴维下标 b 处；不做偶数补齐与交织。

参考 ops-nn 仓 quant/dynamic_mx_quant/tests/assets/golden.py 的 scale_alg=0 路径，
使用纯 torch 接口拼接实现，数值语义逐分支对齐。
"""

# dst_type (int64 枚举) -> (torch dtype 名, emax, exp_bits, mantissa_bits)
# emax: 目标类型最大正则数的指数；exp_bits/mantissa_bits 用于尾数舍入仿真
_DST_TYPE_INFO = {
    35: ("float8_e5m2", 15, 5, 2),
    36: ("float8_e4m3fn", 8, 4, 3),
}

# FP8/E8M0 输出格式: dtype -> (偏置指数 bias, 尾数位宽 mant_bits)
_FP8_FORMATS = {
    torch.float8_e4m3fn: (7, 3),
    torch.float8_e5m2: (15, 2),
    torch.float8_e8m0fnu: (127, 0),
}

_ROUND_MODES = ("rint", "floor", "round")

_E8M0_MAX_BIASED_EXP = 127  # E8M0 偏置指数范围 [-127, 127]


def _pow2(exp: torch.Tensor) -> torch.Tensor:
    """2^exp 的位级精确构造：exp 为 [-127, 127] 内整数或 ±inf/NaN（调用方保证）。

    NPU 的 torch.pow/torch.exp2 对整数指数存在 1 ulp 偏差且次正规数
    flush-to-zero（如 2^-126 在 NPU 上得 0），golden 作为 CPU 参考与
    NPU 候选需逐位一致，故按 IEEE-754 位模式直接构造。
    """
    e = exp.nan_to_num(nan=0.0, posinf=127.0, neginf=-127.0).to(torch.int32)
    bits = (e + 127).clamp(1, 254) << 23  # e ∈ [-126, 127]: 正规数位模式
    bits = torch.where(e == -127, torch.full_like(bits, 0x00400000), bits)  # 2^-127 次正规
    val = bits.contiguous().view(torch.float32)
    val = torch.where(exp == -float("inf"), torch.zeros_like(val), val)  # 2^-inf = 0
    val = torch.where(exp == float("inf"), torch.full_like(val, float("inf")), val)  # 2^inf = inf
    return torch.where(torch.isnan(exp), torch.full_like(val, float("nan")), val)


def _to_fp8(t: torch.Tensor, dtype) -> torch.Tensor:
    """float32 -> FP8/E8M0：位级编码 + view，规避 NPU 不支持的 d2d fp8 cast。

    NPU 上 ``.to(float8_*)`` 走 aclnnInplaceCopy 拷贝路径：E8M0 全平台不支持，
    E4M3/E5M2 在部分平台（如 910B）报 561103。改为整数位运算编码后 ``view``
    （纯元数据操作，无数据搬运），CPU/NPU 数值一致。要求输入值可被目标格式
    精确表示或为 0/次正规/NaN（本算子的量化构造保证 y/mxscale 满足）。
    """
    bias, mant_bits = _FP8_FORMATS[dtype]
    bits = t.contiguous().view(torch.int32)
    abs_bits = bits & 0x7FFFFFFF
    if mant_bits == 0:
        # E8M0 无尾数: 字节 = fp32 偏置指数 (2^e -> e+127, 0 -> 0x00, NaN -> 0xFF)
        return (abs_bits >> 23).to(torch.uint8).view(dtype)
    sign = (bits >> 31) & 1
    mant = abs_bits & 0x7FFFFF
    e = (abs_bits >> 23) - 127
    min_exp = 1 - bias
    # 正规数: 指数字段 e+bias, 尾数字段取 fp32 尾数高位（精确表示保证低位为 0）
    normal = ((e.clamp_min(min_exp) + bias) << mant_bits) | (mant >> (23 - mant_bits))
    # 次正规数: m = value / 2^(min_exp - mant_bits) = (2^23 + mant) >> shift
    sub_shift = (23 - e + min_exp - mant_bits).clamp_min(0)
    subnormal = ((1 << 23) + mant) >> sub_shift
    byte = torch.where(e >= min_exp, normal, subnormal)
    return ((sign << 7) | byte).to(torch.uint8).view(dtype)


def _round_mantissa(t: torch.Tensor, round_mode: str) -> torch.Tensor:
    """按 round_mode 对定点化后的尾数舍入"""
    if round_mode in ("rint", "even"):
        return torch.round(t)  # round-to-nearest-even
    if round_mode in ("round", "nearest"):
        return torch.sign(t) * torch.floor(torch.abs(t) + 0.5)
    if round_mode == "floor":
        return torch.floor(t)
    if round_mode == "ceil":
        return torch.ceil(t)
    if round_mode == "trunc":
        return torch.trunc(t)
    raise RuntimeError(f"Unrecognized round mode: {round_mode}")


def _shared_exp_alg0(amax: torch.Tensor, ele_emax: int) -> torch.Tensor:
    """标准 OCP 算法: shared_exp = floor(log2(amax)) - emax；全零块为 -inf。

    floor(log2(amax)) 直接提取 amax 的 fp32 指数域：正规数下与数学 floor(log2)
    严格一致。不能用 float32 log2 计算——其在二次幂下边界（如 nextafter(2^k, 0)）
    会舍入到整数 k，使 floor 抬高一位、mxscale 偏大 2 倍。次正规 amax 的
    shared_exp 必然低于 -127，由后续 E8M0 钳制兜底（与浮点 log2 路径同结果）。
    """
    abs_bits = amax.contiguous().view(torch.int32) & 0x7FFFFFFF
    exp = ((abs_bits >> 23) - 127).to(torch.float32) - ele_emax
    # NaN/Inf amax 的对数无定义：保持 NaN/Inf 语义（后续 >127 置 NaN，编码 0xFF）
    exp = torch.where(torch.isnan(amax), torch.full_like(exp, float("nan")), exp)
    exp = torch.where(torch.isinf(amax), torch.full_like(exp, float("inf")), exp)
    # 全零块为 -inf（mxscale 编码 0x00）
    return torch.where(amax == 0, torch.full_like(exp, -float("inf")), exp)


def dynamic_mx_quant(
    x: torch.Tensor,
    axis: int = -1,
    round_mode: str = "rint",
    dst_type: int = 36,
    blocksize: int = 32,
):
    """MX 动态量化 (FP8 目标类型)。

    Args:
        x: 输入张量 (float16/bfloat16/float32；评测框架会以 float64 传入，同样接受)
        axis: 量化发生的轴，[-D, D-1]
        round_mode: 舍入模式，FP8 目标类型仅支持 "rint"
        dst_type: 目标类型枚举，35=float8_e5m2, 36=float8_e4m3fn
        blocksize: 每块元素个数，32 的倍数且不超过 1024

    Returns:
        y: 量化后张量 (dst_type 对应 dtype，shape 与 x 一致)
        mxscale: 每块量化尺度 (torch.float8_e8m0fnu，rank 与 x 一致，
            量化轴维为 ceil(x.shape[axis]/blocksize)，其余维度与 x 一致)
    """
    if not x.is_floating_point():
        raise RuntimeError(f"Unsupported input dtype: {x.dtype}")
    if dst_type not in _DST_TYPE_INFO:
        raise RuntimeError(
            f"Unsupported dst_type: {dst_type} (支持 35=float8_e5m2, 36=float8_e4m3fn)"
        )
    dtype_name, ele_emax, exp_bits, mant_bits = _DST_TYPE_INFO[dst_type]
    out_dtype = getattr(torch, dtype_name)

    dim = x.dim()
    if axis < 0:
        axis += dim
    if not 0 <= axis < dim:
        raise RuntimeError(f"axis {axis} out of range for rank {dim}")
    if blocksize <= 0 or blocksize % 32 != 0 or blocksize > 1024:
        raise RuntimeError(f"blocksize must be a multiple of 32 and <= 1024, got {blocksize}")
    if x.dtype == torch.float32 and blocksize != 32:
        raise RuntimeError("float32 input only supports blocksize=32")
    if round_mode not in _ROUND_MODES:
        raise RuntimeError(f"round_mode must be one of {_ROUND_MODES}, got {round_mode}")

    # 统一到 float32 计算精度，并把量化轴换到末尾按尾轴分块处理
    xf = x.to(torch.float32)
    tail_axis = axis == dim - 1
    xt = xf.movedim(axis, -1) if not tail_axis else xf

    n = xt.shape[-1]
    n_blocks = (n + blocksize - 1) // blocksize
    pad_len = n_blocks * blocksize - n
    if pad_len > 0:
        xt = torch.nn.functional.pad(xt, (0, pad_len))  # 不足一块按 0 补齐
    blocks = xt.unflatten(-1, (n_blocks, blocksize))

    # 每块绝对值最大值 -> 共享指数 -> E8M0 值域裁剪
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    share_exp = _shared_exp_alg0(amax, ele_emax)
    share_exp = torch.where(
        share_exp > _E8M0_MAX_BIASED_EXP,
        torch.full_like(share_exp, float("nan")),
        share_exp,
    )
    share_exp = torch.where(share_exp < -_E8M0_MAX_BIASED_EXP, torch.full_like(share_exp, -float(_E8M0_MAX_BIASED_EXP)), share_exp)
    mxscale_val = _pow2(share_exp)  # [..., n_blocks, 1]

    # 元素级量化：去尺度 -> 私有指数(截断到最小正则指数, 仿真 subnormal) -> 定点舍入 -> 还原
    ret = blocks / mxscale_val
    private_exp = torch.floor(torch.log2(torch.abs(ret) + (ret == 0).to(torch.float32)))
    min_exp = -(2 ** (exp_bits - 1)) + 2
    private_exp = private_exp.clamp_min(min_exp)
    ret = ret / _pow2(private_exp) * (2**mant_bits)
    ret = _round_mantissa(ret, round_mode)
    ret = ret / (2**mant_bits) * _pow2(private_exp)
    max_norm = float(torch.finfo(out_dtype).max)
    ret = ret.clamp(-max_norm, max_norm)
    ret = torch.nan_to_num(ret, nan=0.0)  # NaN -> 0

    # 去掉块内 padding 并把轴换回原位
    y = ret.flatten(-2)[..., :n]
    if not tail_axis:
        y = y.movedim(-1, axis)
    y = _to_fp8(y, out_dtype)

    # mxscale 布局：逐块尺度张量，量化轴维 = n_blocks，其余维度与 x 一致（无补偶/配对/交织）
    scale_val = mxscale_val.squeeze(-1)
    if not tail_axis:
        scale_val = scale_val.movedim(-1, axis)
    mxscale = _to_fp8(scale_val, torch.float8_e8m0fnu)

    return y, mxscale
