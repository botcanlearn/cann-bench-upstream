# Trsv 算子 API 描述

## 1. 算子简介

Trsv（Triangular matrix Solve）求解三角线性系统 `op(A) * x = b`，是 BLAS 中的核心例程之一（对应 cuBLAS 的 `trsv` / LAPACK 的 `trtrs`）。ops-blas 提供单精度实数（`aclblasStrsv`，FP32）与单精度复数（`aclblasCtrsv`，complex64）两个接口，解向量原地覆写右端向量 b。

**主要应用场景**：
- 线性方程组求解（LU/Cholesky 分解后的三角回代）
- 最小二乘问题、矩阵求逆的子步骤
- 科学计算与信号处理中的稠密线性代数流水线

**算子特征**：
- 难度等级：L3（Contraction）—— 多属性组合（2×3×2）、三角依赖的串行求解段、panel 分块与多核协同
- 双输入（A、x）、原地输出（x 被覆写）
- 支持 FP32 与 complex64 两种精度

## 2. 算子定义

### 数学公式

$$
op(A) \cdot x = b
$$

其中：
- `uplo=UPPER`：仅引用 A 的上三角；`uplo=LOWER`：仅引用 A 的下三角
- `trans` 决定 op(A)：`N` → A；`T` → Aᵀ；`C` → Aᴴ（复数共轭转置，FP32 下与 T 等价）
- `diag=UNIT`：op(A) 的对角元视为 1（A 中存储的对角值被忽略）；`diag=NON_UNIT`：对角元从 A 读取

求解方式为三角代换：LOWER 为前向代换（x_i = (b_i − Σ_{j<i} A_ij·x_j) / A_ii），UPPER 为后向代换。

### 特殊情况

| 输入 | 行为 |
|------|------|
| n = 0 | 空操作，直接返回成功 |
| diag = UNIT | 对角元固定为 1，A 对角存储值不参与计算 |
| incx < 0 | x 反向存储（BLAS 指针约定）：向量元素 i 位于 x[(n-1-i)·\|incx\|] |
| 非 uplo 三角 | 仅被引用三角参与计算，另一三角内容不影响结果 |
| 非法参数（uplo/trans/diag 越界、lda < max(1,n)、incx = 0、n < 0、空指针） | 返回 ACLBLAS_STATUS_INVALID_VALUE |

## 3. 接口规范

### 算子原型

```python
cann_bench.trsv(Tensor A, Tensor x, str uplo, str trans, str diag, int incx) -> Tensor x
```

底层 C API（摘自 ops-blas README）：

```cpp
aclblasStatus_t aclblasStrsv(aclblasHandle_t handle, aclblasFillMode_t uplo,
    aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const float *A, int lda, float *x, int incx);
aclblasStatus_t aclblasCtrsv(aclblasHandle_t handle, aclblasFillMode_t uplo,
    aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const aclblasComplex* A, int lda, aclblasComplex* x, int incx);
```

### 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|----------|------|
| uplo | 输入 | ACLBLAS_UPPER(121) 上三角；ACLBLAS_LOWER(122) 下三角 |
| trans | 输入 | ACLBLAS_OP_N(111)；ACLBLAS_OP_T(112) Aᵀ；ACLBLAS_OP_C(113) Aᴴ |
| diag | 输入 | ACLBLAS_NON_UNIT(131)；ACLBLAS_UNIT(132) |
| n | 输入 | 矩阵阶数，n ≥ 0，n = 0 空操作 |
| A | 输入 | n×lda 三角矩阵，行主序存储，仅 uplo 三角被访问 |
| lda | 输入 | A 的前导维，lda ≥ max(1, n) |
| x | 输入/输出 | 输入存右端 b，输出原地覆写为解；缓冲区长度 (n-1)·\|incx\|+1，按 incx 步长访问 |
| incx | 输入 | x 的存储增量，incx ≠ 0（可正可负） |

注：`n`、`lda` 为底层 C API 形参，cann_bench python 接口（proto schema）不单独暴露——n 由 A.shape[0] 与 x 长度隐含，lda 由 A.shape[1] 隐含。cases.yaml/csv 的 attrs 键名仅与 proto 声明的 uplo/trans/diag/incx 对应（contributing.md §4.2）。

### 数据类型

| 输入 dtype（A / x） | 输出 dtype（原地覆写 x） | 对应 C API |
|---------------------|--------------------------|------------|
| float32             | float32                  | aclblasStrsv |
| complex64           | complex64                | aclblasCtrsv |

A 与 x 的 dtype 必须一致；输出原地覆写 x，输出 dtype 与输入一致。当前打榜用例集全部为 float32（complex64 覆盖缺口见 §8）。

## 4. 约束说明（摘自 ops-blas README）

- n ≥ 0，n == 0 时为空操作直接返回成功
- uplo 必须为 ACLBLAS_UPPER 或 ACLBLAS_LOWER
- trans 必须为 ACLBLAS_OP_N、ACLBLAS_OP_T 或 ACLBLAS_OP_C
- diag 必须为 ACLBLAS_NON_UNIT 或 ACLBLAS_UNIT
- lda ≥ max(1, n)
- incx ≠ 0（可正可负）
- n > 0 时 A、x 不可为 nullptr

## 5. 标准 Golden 代码

Torch Golden 参考实现（与 `golden.py` 一致，求解 `op(A) * sol = b`，原地覆写返回 x 缓冲区）：

```python
import torch


def trsv(
    A: torch.Tensor,
    x: torch.Tensor,
    uplo="LOWER",
    trans="N",
    diag="NON_UNIT",
    incx: int = 1,
) -> torch.Tensor:
    """求解 op(A) * sol = b，返回覆写后的 x 缓冲区。

    Args:
        A: n×lda 三角矩阵（行主序），仅 uplo 指定三角被引用
        x: 右端向量 b 的缓冲区，长度 (n-1)*|incx|+1，按 incx 步长访问
        uplo: UPPER/LOWER；trans: N/T/C；diag: NON_UNIT/UNIT
        incx: x 的存储增量（非零，可负）
    """
    incx = int(incx)
    n = int(A.shape[0])
    if n == 0:
        return x.clone()

    # op(A)：N -> A；T -> A^T；C -> A^H（实数 conj 为恒等）
    # uplo 描述 A 的存储三角；先清掉另一侧再转置（转置后三角自然翻转）
    M = A[:, :n]
    M = M.triu() if uplo == "UPPER" else M.tril()
    if trans == "T":
        M = M.t()
    elif trans == "C":
        M = M.t().conj()
    is_upper = (uplo == "UPPER") ^ (trans in ("T", "C"))

    # 按 incx 步长取出右端向量
    # incx<0 为 BLAS 反向存储约定: vec[i] = x[(n-1-i)*|incx|]
    # （torch 切片不支持负步长, 用 flip(0)+正步长等价实现）
    if incx > 0:
        vec = x[::incx]
    else:
        vec = x.flip(0)[::(-incx)]

    # 升精度计算（FP32/complex64 求解以 FP64/complex128 作参考，降低 golden 自身误差）
    compute_dtype = torch.complex128 if A.is_complex() else torch.float64
    sol = torch.linalg.solve_triangular(
        M.to(compute_dtype),
        vec.to(compute_dtype).unsqueeze(-1),
        upper=is_upper,
        unitriangular=(diag == "UNIT"),
        left=True,
    ).squeeze(-1).to(x.dtype)

    # 原地覆写语义：仅 strided 位置写入解，其余位置保持输入值
    out = x.clone()
    if incx > 0:
        out[::incx] = sol
    else:
        out = out.flip(0)
        out[::(-incx)] = sol
        out = out.flip(0)
    return out
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

# LOWER / N / NON_UNIT：前向代换，incx=1 连续访问
n = 1024
A = torch.randn(n, n, dtype=torch.float32, device="npu")  # 仅 uplo 指定三角被引用
x = torch.randn(n, dtype=torch.float32, device="npu")     # 输入存右端 b，输出原地覆写为解
sol = cann_bench.trsv(A, x, uplo="LOWER", trans="N", diag="NON_UNIT", incx=1)

# UPPER / T / UNIT：后向代换 + incx=2 步长访问，x 缓冲区长度 (n-1)*|incx|+1
n = 512
A = torch.randn(n, n, dtype=torch.float32, device="npu")
x = torch.randn((n - 1) * 2 + 1, dtype=torch.float32, device="npu")
sol = cann_bench.trsv(A, x, uplo="UPPER", trans="T", diag="UNIT", incx=2)
```

## 7. 实现参考要点（arch22 / DAV_2201 实测源码行为）

以下要点来自 `ops-blas/blas/trsv/arch22/` 的 host/kernel 源码，供实现与评审核对：

- **A 为行主序 n×lda**：kernel `CopyIn` 中 OP_N 按 `aOffset = row*lda + col`、OP_T 按 `aOffset = col*lda + row` 取值（`strsv_kernel.cpp`）
- **incx 由 host 侧 gather/scatter**：进入 kernel 前 `xHost[i] = x[i*incx]`，完成后 `x[i*incx] = xHost[i]`（`strsv_host.cpp`）；kernel 内按连续向量处理
- **strsv（FP32）**：单核（numBlocks=1），按 8 元素块（32B/4B）分块 tiling
- **ctrsv（complex64）**：panel 分块算法（b=64），对角块 UB 驻留串行求解 + off-diagonal 向量化更新；多核 useCoreNum=min(8, nBlocks)，跨核 SyncAll 屏障同步
- **三角求解段沿对角线严格串行**（n 步，数学约束），是 arch22 上的主要性能瓶颈

## 8. 评测注意与用例设计说明

- **原地覆写语义**：解只写入按 incx 步长访问的位置，x 缓冲区其余位置保持输入值不变。golden 输出与 kernel 输出均须满足此语义
- **数值稳定性与值域设计**：随机三角矩阵求解的解量级随 n 指数放大（实测 n=64、A∈[1,10] 时解已达 ~1e9，n≥128 溢出）。因此本任务用例采用如下构造：
  - **n ≥ 63 的用例一律 diag=UNIT、A ∈ [-0.01, 0.01]**：等效矩阵为 I+E（E 为微小随机矩阵），条件数 O(1)，fp32 精度实测误差 ≤ 1.1e-5（阈值 0.005），远优于阈值
  - **NON_UNIT 仅在小 n（≤5）覆盖**：此时解量级可控
  - 大 n 用例的数值不影响性能测量（固定计算模式），UNIT + 微扰构造不影响性能口径
- **trans=C 的覆盖设计**：README 明确 FP32 下 OP_C 与 OP_T 等价（实数 Aᴴ=Aᵀ）。case 9（LOWER/C/NON_UNIT）、case 10（UPPER/C/NON_UNIT, incx=2, 唯一步长≠1 的 C 路径）、case 12（UPPER/C/UNIT）覆盖 aclblasStrsv 接受 C(113) 的 dispatch 路径，且实数语义可精确映射到 e2e 参考实现
- **complex64（aclblasCtrsv）覆盖缺口说明**：当前栈（CANN 9.0.0 + torch_npu）不存在 NPU 可执行的复数三角求解参考路径（solve_triangular 报 161002；复数 mm/tril 缺失；linalg.solve 复数为 host 回退，128×128 单次 33.7ms），故打榜用例集暂全部为 float32。proto 仍声明 complex64 能力，复数路径的精度/性能考核待参考实现可用后补充用例
- **基线可测性**：20 条用例的 baseline_perf_us 均为 TTK e2e 真机实测（20/20 PASS），t_hw_us 按 hap-ascend-910b2-v2 skill 的 roofline 方法计算（全部 hbm_read 瓶颈）
- **precision_thresholds**：float32=0.005、complex64=0.01（三角求解为串行除法+累加，误差随 n 增长，参考除法类算子的宽松阈值惯例）
- **基线口径**：README 性能表（TC_PF_1001~1003）为 ctrsv arch22 实测，arch22 相对 GPU cuBLAS 标杆存在已确认的架构代际差距（约 107~140×，详见 ops-blas README 性能说明），打榜基线评估时需注意口径
