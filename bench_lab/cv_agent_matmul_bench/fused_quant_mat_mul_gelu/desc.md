# FusedQuantMatMul A8W8 per-token/per-channel GELU_ERF BF16 算子描述

## 1. 算子简介

FusedQuantMatMul 将量化矩阵乘、反量化、浮点 bias 和 GELU 融合在一个算子中。本 benchmark 选取大语言模型和 Transformer 线性层常用的 A8W8 路径：`x1` 与 `x2` 为 INT8，矩阵乘采用 INT32 累加语义，随后使用 per-token `x1Scale` 和 per-channel `x2Scale` 恢复浮点结果，加上 BF16 bias，执行 `gelu_erf` 并输出 BF16。

**主要应用场景**：

- Transformer 前馈网络中的量化线性层与 GELU 融合
- 大语言模型和编码器模型的 decode、batch-decode 与 prefill
- 需要同时使用激活 per-token scale 和权重 per-channel scale 的 A8W8 推理

**算子特征**：

- 难度等级：L3（Quantized MatMul + Dequantization + Bias + Nonlinear Activation）
- 输入矩阵：INT8 `x1[M,K]` 与 INT8 `x2[K,N]`
- 反量化：FP32 `x1Scale[M]` 与 BF16 `x2Scale[N]`
- 融合后处理：BF16 浮点 bias 和 `gelu_erf`
- 输出：BF16 `out[M,N]`
- 实现形态：同一套公开接口与通用 Kernel 处理不同 `M/K/N`，Host Tiling 根据运行时 shape 和硬件资源自适应生成切分参数

本 benchmark 固定 2D ND、非转置、A8W8、x1 per-token、x2 per-channel、BF16 浮点 bias、`gelu_erf` 和 BF16 输出，不覆盖其他数据类型、量化模式或融合操作。

## 2. 算子定义

### 数学公式

INT8 输入矩阵乘使用 INT32 累加语义：

$$
acc = \operatorname{MatMul}_{INT32}(x1, x2)
$$

`x2Scale` 沿 N 维广播，`x1Scale` 沿 M 维广播，再按浮点 bias 路径相加：

$$
qbmmout
=
\operatorname{FP32}(acc)
\odot x2Scale[None,:]
\odot x1Scale[:,None]
+ bias[None,:]
$$

本 benchmark 固定使用基于误差函数的 GELU：

$$
gelu\_erf(z)
=
\frac{z}{2}
\left(
1+\operatorname{erf}\left(\frac{z}{\sqrt{2}}\right)
\right)
$$

最终输出为：

$$
out = \operatorname{BF16}\left(gelu\_erf(qbmmout)\right)
$$

### 步骤说明

1. **量化矩阵乘**：将 INT8 `x1` 和 `x2` 按 INT32 累加语义计算为 `[M,N]`。
2. **per-channel 反量化**：将 `[N]` 的 BF16 `x2Scale` 广播到 N 维。
3. **per-token 反量化**：将 `[M]` 的 FP32 `x1Scale` reshape 为 `[M,1]` 并广播到 M 维。
4. **浮点 bias**：将 `[N]` 的 BF16 `bias` 沿 M 维广播后相加。该顺序属于官方定义的浮点 bias 路径，不是 INT32 bias 的 `(matmul + bias) * scale` 路径。
5. **GELU 与输出转换**：按 `gelu_erf` 计算激活值并转换为 BF16 `out`。

## 3. 接口规范

### 算子原型

```python
cann_bench.fused_quant_mat_mul(
    Tensor x1,
    Tensor x2,
    Tensor x1Scale,
    Tensor x2Scale,
    Tensor bias,
) -> Tensor out
```

官方 aclnn 接口中的 `fusedOpType` 由适配层固定为 `"gelu_erf"`。本 benchmark 也固定非转置和 BF16 输出，因此不将这些固定选择暴露为可变属性。

### 输入参数

| 参数 | 类型 | Shape | dtype | 描述 |
|---|---|---|---|---|
| `x1` | Tensor（必选） | `[M,K]` | INT8 | 量化激活矩阵 |
| `x2` | Tensor（必选） | `[K,N]` | INT8 | 量化权重矩阵 |
| `x1Scale` | Tensor（必选） | `[M]` | FLOAT32 | x1 的 per-token scale，每个 M 行一个缩放因子 |
| `x2Scale` | Tensor（必选） | `[N]` | BFLOAT16 | x2 的 per-channel scale，每个 N 列一个缩放因子 |
| `bias` | Tensor（必选） | `[N]` | BFLOAT16 | 反量化之后相加的浮点 bias |

### 输出

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `out` | `[M,N]` | BFLOAT16 | 反量化、bias 和 `gelu_erf` 融合后的结果 |

### 数据类型组合

| `x1` | `x2` | `x1Scale` | `x2Scale` | `bias` | `out` |
|---|---|---|---|---|---|
| INT8 | INT8 | FLOAT32 | BFLOAT16 | BFLOAT16 | BFLOAT16 |

### 规则与约束

- `x1` 和 `x2` 必须是非空的二维 ND Tensor，且本 benchmark 固定为连续、非转置输入。
- `x1.shape == [M,K]`，`x2.shape == [K,N]`，三个逻辑维度都必须大于 0。
- `x1Scale` 必须是一维 `[M]`，本 benchmark 不覆盖缺少 x1 per-token scale 的路径。
- `x2Scale` 必须是一维 `[N]`，本 benchmark 不覆盖 `[1]` per-tensor scale。
- `bias` 必须是一维 `[N]`，本 benchmark 不覆盖无 bias、INT32 bias 或三维 batch bias。
- `x1` 和 `x2` 的 K 维必须相等。
- 在当前非转置 ND 路径中，`x1` 的最后一维 K 和 `x2` 的最后一维 N 均不得超过 65535。
- 本 benchmark 固定 `gelu_erf` 与 BF16 输出，不覆盖 `gelu_tanh`、FP16 输出、INT4、batch、转置、offset 或分组量化路径。

### 支持范围

| 维度 / 参数 | cases 覆盖 | 备注 |
|---|---:|---|
| `M` | 1～2048 | 单 token、batch-decode、prefill 与 M 尾块 |
| `K` | 64～8192 | 常见隐藏维及 `K=65/769` 非对齐尾块 |
| `N` | 33～28672 | 常见 FFN 宽度及 `N=33/3073` 非对齐尾块 |
| 输入 dtype | INT8 / INT8 / FP32 / BF16 / BF16 | 所有 case 使用同一接口组合 |
| 输出 dtype | BFLOAT16 | 所有 case 固定 |
| `x1` 测试值域 | `[-1,1]` | signed 小整数，覆盖负数、零和正数 |
| `x2` 测试值域 | `[-5,5]` | signed 小整数，扩大权重侧取值覆盖 |
| `x1Scale` 测试值域 | `[0,1]` | 非负 per-token scale |
| `x2Scale` 测试值域 | `[0,1]` | 非负 per-channel scale |
| `bias` 测试值域 | `[-5,5]` | 覆盖正、负 bias |

上述值域是 benchmark 的测试数据生成策略，参考已有算子库测试中对不同输入分别设置的范围，用于覆盖符号、零值和非负 scale，同时控制大 K 矩阵乘的数值规模。它们不是 aclnn 接口文档规定的输入值域约束。

### 自适应 Tiling 要求

所有 case 共用上述接口和计算模板。生成的 AscendC 算子必须根据运行时 `M/K/N` 和目标硬件资源自适应规划矩阵基本块、核间任务、UB/L1/L2 使用及尾块处理，并由同一份通用 Kernel 消费相应 tiling 数据。实现可以采用不同于算子库的切分策略，不要求复刻其内部 tiling 参数，但不得通过枚举 benchmark case shape 并写死对应配置来代替通用 tiling。

## 4. 精度要求

采用 [CANN 生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)。

对 BF16 `out` 统计平均相对误差 MERE 和最大相对误差 MARE：

$$
\operatorname{MERE}
=
\operatorname{avg}
\left(
\frac{|actual-golden|}{|golden|+10^{-7}}
\right)
$$

$$
\operatorname{MARE}
=
\max
\left(
\frac{|actual-golden|}{|golden|+10^{-7}}
\right)
$$

`out` 是普通 BF16 浮点输出，采用 `output_dtype` 规则：

| 输出 | dtype | Threshold | 通过条件 |
|---|---|---:|---|
| `out` | BFLOAT16 | `2^-7` | `MERE < 2^-7` 且 `MARE < 10 * 2^-7` |

## 5. 标准 Golden 代码

```python
import torch
from torch.nn.functional import gelu as torch_gelu


def fused_quant_mat_mul(
    x1: torch.Tensor,
    x2: torch.Tensor,
    x1Scale: torch.Tensor,
    x2Scale: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """FusedQuantMatMul 的 INT8、per-channel、gelu_erf 路径 Golden。

    Args:
        x1: shape 为 [M, K]、dtype 为 torch.int8 的 Tensor。
        x2: shape 为 [K, N]、dtype 为 torch.int8 的 Tensor。
        x1Scale: shape 为 [M]、dtype 为 torch.float32 的 per-token scale。
        x2Scale: shape 为 [N]、dtype 为 torch.bfloat16 的
            per-channel scale。
        bias: shape 为 [N]、dtype 为 torch.bfloat16 的 Tensor。

    Returns:
        shape 为 [M, N]、dtype 为 torch.bfloat16 的 Tensor。

    计算公式:
        qbmmout = (x1 @ x2) * x2Scale * x1Scale + bias
        out = gelu_erf(qbmmout)
    """
    # PyTorch 的 INT8 矩阵乘会返回 INT8，无法表达量化矩阵乘的
    # INT32 累加语义，因此这里将两个输入转换为 INT32。
    qbmmout = torch.matmul(x1.to(torch.int32), x2.to(torch.int32))

    # 将累加结果转换为 FP32，以 FP32 执行后续缩放、bias 相加和
    # gelu_erf。若直接计算 INT32 * BF16，PyTorch 会产生 BF16
    # 中间结果并发生过早舍入。
    qbmmout = qbmmout.to(torch.float32) * x2Scale

    # x1Scale 已经是 FP32；x2Scale 和 bias 虽然是 BF16，但与 FP32
    # 中间结果运算时会按 PyTorch 类型提升规则参与 FP32 计算。
    qbmmout = qbmmout * x1Scale.unsqueeze(-1)
    qbmmout = qbmmout + bias

    out = torch_gelu(qbmmout, approximate="none")

    # aclnnFusedQuantMatmul 这条已选路径的接口输出 dtype 为 BF16。
    return out.to(torch.bfloat16)
```

上述代码表达当前 benchmark 的计算语义与输出契约：

- INT8 输入转换为 INT32 是为了在 PyTorch 中表达 INT8 矩阵乘的 INT32 累加语义；
- 矩阵乘结果转换为 FP32，使 BF16 `x2Scale` 和 `bias` 在后处理阶段不会造成过早的 BF16 舍入；
- `x1Scale.unsqueeze(-1)` 只负责将 `[M]` 变为 `[M,1]` 以完成广播；
- `torch_gelu(..., approximate="none")` 对应 `gelu_erf`；
- 最后的 BF16 转换属于公开输出契约。

Golden 用于定义可观察的数学语义，不要求生成算子逐指令复制算子库的中间计算和切分方式。

## 6. 额外信息

### Golden 调用示例

```python
import torch

from golden import fused_quant_mat_mul

M, K, N = 8, 4096, 11008
x1 = torch.randint(-1, 2, (M, K), dtype=torch.int8)
x2 = torch.randint(-5, 6, (K, N), dtype=torch.int8)
x1_scale = torch.rand((M,), dtype=torch.float32)
x2_scale = torch.rand((N,), dtype=torch.float32).to(torch.bfloat16)
bias = (torch.rand((N,), dtype=torch.float32) * 10 - 5).to(torch.bfloat16)

out = fused_quant_mat_mul(x1, x2, x1_scale, x2_scale, bias)

assert out.shape == (M, N)
assert out.dtype == torch.bfloat16
```

### Case 设计

`cases.yaml` 与 `cases.csv` 一一对应，共包含 20 个使用相同接口和数据类型组合的正向 case：

- 3 个基础或非对齐尾块 case；
- 9 个 LLM decode 或 batch-decode case；
- 8 个 LLM prefill、prefill tail 或长序列 case；
- 覆盖 M/K/N 对齐与非对齐情况，包括 `M=7,K=65,N=33`、`M=17,K=769,N=3073` 和 `M=257`；
- 覆盖从单 token 到 `M=2048`、从 `K=64` 到 `K=8192`、从 `N=33` 到 `N=28672` 的工作负载；
- `value_range` 按输入名分别声明：`x1=[-1,1]`、`x2=[-5,5]`、`x1Scale/x2Scale=[0,1]`、`bias=[-5,5]`，不会将矩阵输入范围错误套用到 scale 和 bias；
- 所有 case 共用同一计算模板接口，用于验证生成算子的运行时自适应 tiling、尾块正确性和不同规模下的性能。

案例文件不携带或约束算子库内部的 tiling 选择。生成实现可以自行设计自适应切分，验收以接口、计算语义、输出精度及性能为准。

### 实现对齐依据

- `docs/aclnnFusedQuantMatmul.md`：浮点 bias、双 scale 与 `gelu_erf` 的接口公式和 dtype/shape 约束
- `op_kernel/fused_quant_mat_mul.cpp`：FusedQuantMatMul Kernel 入口
- `quant_batch_matmul_v3_pertoken_basic.h`：INT8 矩阵乘与 per-token 后处理路径
- `quant_batch_matmul_v3_basic_epilogue.h`：FP32 scale、浮点 bias、GELU_ERF 和 BF16 写回路径
- `tests/st/aclnnFusedQuantMatmul/executor_aclnnFusedQuantMatmul.py`：PyTorch 精度对比参考
- `tests/st/aclnnFusedQuantMatmul/arch22_atk_aclnnFusedQuantMatmul.json`：按输入分别设置测试值域的参考

### 参考资料

- [CANN 9.0 aclnnFusedQuantMatmul 文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/aolapi/context/ops-nn/aclnnFusedQuantMatmul.md)
