# DynamicMxQuant 算子 API 描述

## 1. 算子简介

MX (microscaling) 动态量化：在给定轴上按 blocksize 分块，为每块计算 E8M0 共享尺度 mxscale，块内元素除以 mxscale 后舍入到目标低精度类型。

**主要应用场景**：
- 大语言模型推理中 FP8/FP4 低精度计算的在线量化（MXFP8 GEMM 的前处理）
- KV Cache / 激活值的块级压缩存储
- OCP MX 规范（microscaling formats）在 NPU 上的落地

**算子特征**：
- 难度等级：L3（FusedComposite）
- 单输入双输出，融合块内归约（amax）、共享指数计算与逐元素量化
- 输出 y 为 FP8（由 dst_type 指定），mxscale 恒为 E8M0

## 2. 算子定义

### 数学公式

采用标准 OCP MX 算法——将输入 x 在 axis 维上按 k=blocksize 分组，一组 k 个数 $\{V_i\}_{i=1}^{k}$ 动态量化为 $\{mxscale, \{P_i\}_{i=1}^{k}\}$：

$$
shared\_exp = \lfloor \log_2(\max_i(|V_i|)) \rfloor - emax \\
mxscale = 2^{shared\_exp} \\
P_i = cast\_to\_dst\_type(V_i / mxscale,\ round\_mode)
$$

- 量化后的 $P_i$ 按 $V_i$ 的位置组成输出 y，mxscale 按 axis 维的分组组成输出 mxscale
- emax 为目标数据类型最大正则数的指数；全零块 $shared\_exp = -\infty$（mxscale 编码为 0x00）
- shared_exp 超出 E8M0 范围时：大于 127 置为 NaN，小于 -127 截断到 -127
- 仅支持此标准算法，不提供其他 scale 计算方法

emax 对照：

| DataType | emax |
| :------: | :--: |
| FLOAT8_E4M3FN | 8 |
| FLOAT8_E5M2 | 15 |

## 3. 接口规范

### 算子原型

```python
cann_bench.dynamic_mx_quant(Tensor x, int axis, str round_mode, int dst_type, int blocksize) -> (Tensor y, Tensor mxscale)
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 待量化数据，1-7 维 ND |
| axis | int | 必选 | 量化发生的轴，[-D, D-1]，D 为 x 的维数 |
| round_mode | str | "rint" | 舍入模式；FP8 目标类型仅支持 "rint" |
| dst_type | int | 必选 | 目标类型枚举：35=float8_e5m2，36=float8_e4m3fn |
| blocksize | int | 必选 | 每块元素个数，32 的倍数且不超过 1024；x 为 float32 时必须为 32 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 与输入 x 相同 | dst_type 对应类型 | 量化后的张量 |
| mxscale | rank 与 x 一致；量化轴维 = ceil(x.shape[axis]/blocksize)，其余维度与 x 一致 | float8_e8m0 | 每块量化尺度 |

### 数据类型

| 输入 x dtype | y dtype | mxscale dtype |
|-------------|---------|---------------|
| float16 | float8_e4m3fn / float8_e5m2 | float8_e8m0 |
| bfloat16 | float8_e4m3fn / float8_e5m2 | float8_e8m0 |
| float32 | float8_e4m3fn / float8_e5m2 | float8_e8m0 |

### 规则与约束

- mxscale 的 shape 约束：`rank(mxscale) = rank(x)`；`axis_change = axis if axis >= 0 else axis + rank(x)`；`mxscale.shape[axis_change] = ceil(x.shape[axis] / blocksize)`；其他维度与 x 一致
- mxscale 为逐块尺度张量，第 b 块的尺度直接存于量化轴维下标 b 处；不做偶数补齐、不引入尾维 2、不涉及交织处理
- axis 维不是 blocksize 整数倍时，块尾不足部分按 0 补齐
- x 为 float32 时仅支持 blocksize=32，且量化轴的维度不能小于 32
- FP4 目标类型（dst_type=40/41）不在本任务支持范围：torch CPU 侧无法表示 packed FP4（float4_e2m1fn_x2 为双元素打包且不支持 cast），golden 与精度比对均无法构造；FP8 场景不受影响
- 仅支持标准 OCP MX 量化算法，不支持其他 scale 计算方法
- round_mode：FP8 目标类型仅支持 "rint"（round-to-nearest-even）
- 量化确定性：同一输入的输出确定

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `rank(x)`（输入维度数） | 1 ~ 7 | cases.csv 实测 2 ~ 5 维 |
| 各维度大小 `dim_i` | 1 ~ 65536 | cases.csv 实测 2 ~ 16384 |
| 量化轴维度（分块粒度） | 1 ~ 65536（float32 须 ≥ 32） | cases.csv 实测 64 ~ 16384；不足一块按 0 补齐 |
| `axis` | [-D, D-1] | cases.csv 实测 -1、0、1、2、3（含非尾轴场景） |
| `dst_type` | {35, 36} | 40/41（FP4）不支持；cases.csv 中 40/41 仅用于 perf 基线用例，精度评测不支持 |
| `blocksize` | {32, 64, ..., 1024} | 32 的倍数；float32 输入固定 32；cases.csv 实测 32 / 64 |
| `round_mode` | {"rint"} | FP8 仅支持 rint；cases.csv 实测 rint/floor/round（floor/round 仅 FP4 perf 用例） |
| 张量总元素数 | 1 ~ 2^30 | cases.csv 实测最大约 17M |

## 4. 精度要求

采用[生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)进行验证。

**误差指标**：

1. 平均相对误差（MERE）：采样点中相对误差平均值

   $$
   \text{MERE} = \text{avg}(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

2. 最大相对误差（MARE）：采样点中相对误差最大值

   $$
   \text{MARE} = \max(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

**通过标准**：

| 数据类型 | FLOAT16 | BFLOAT16 | FLOAT32 | HiFLOAT32 | FLOAT8 E4M3 | FLOAT8 E5M2 |
|----------|---------|----------|---------|-----------|-------------|-------------|
| **通过阈值(Threshold)** | 2^-10 | 2^-7 | 2^-13 | 2^-11 | 2^-3 | 2^-2 |

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过。

**E8M0 mxscale 精度要求**：mxscale 为 2 的整数次幂，任何指数偏差（≥1 个指数）的相对误差均 ≥ 0.5，远超默认相对误差阈值，判不通过；编码一致（相对误差 0）方可通过。y 输出仍按上表 FP8 阈值判定。

## 5. 标准 Golden 代码

```python
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
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

# 尾轴 MXFP8 量化（LLM 推理典型场景）
x = torch.randn(1024, 4096, dtype=torch.bfloat16, device="npu")
y, mxscale = cann_bench.dynamic_mx_quant(
    x, axis=-1, round_mode="rint", dst_type=36, blocksize=32
)
# y.shape = (1024, 4096),         y.dtype = torch.float8_e4m3fn
# mxscale.shape = (1024, 128),    mxscale.dtype = torch.float8_e8m0fnu

# 非尾轴量化
x = torch.randn(64, 8192, dtype=torch.float16, device="npu")
y, mxscale = cann_bench.dynamic_mx_quant(
    x, axis=0, round_mode="rint", dst_type=36, blocksize=32
)
# y.shape = (64, 8192); mxscale.shape = (2, 8192)

# 下游反量化（典型用法）：块内乘回 mxscale
scale = mxscale.to(torch.float32).repeat_interleave(32, dim=-1)
dequant_x = y.to(torch.float32) * scale
```
