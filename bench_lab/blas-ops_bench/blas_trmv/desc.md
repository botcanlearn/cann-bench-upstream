# Trmv 算子 API 描述

## 1. 算子简介

Trmv（Triangular Matrix-Vector Multiplication）实现三角矩阵与向量的乘法 `x := op(A) * x`，是 BLAS 核心例程之一（对应 cuBLAS 的 `trmv` / LAPACK 的 `trmv`）。ops-blas 提供单精度实数（`aclblasStrmv`，FP32）与单精度复数（`aclblasCtrmv`，complex64）两个接口，结果原地覆写输入向量。

**主要应用场景**：
- 三角线性方程组求解的迭代子步（与 Trsv 配合的矩阵-向量乘部分）
- 稀疏/结构化线性代数中的三角变换
- 数值算法中的前向/后向传播式计算

**算子特征**：
- 难度等级：L2（Contraction）—— 属性组合与分块访存，但输出各元素相互独立（无三角求解的串行依赖）
- 双输入（A、x）、原地输出（x 被覆写）
- 支持 FP32 与 complex64 两种精度

## 2. 算子定义

### 数学公式

$$
x := op(A) \cdot x
$$

其中：
- `uplo=UPPER`：仅引用 A 的上三角；`uplo=LOWER`：仅引用 A 的下三角
- `trans` 决定 op(A)：`N` → A；`T` → Aᵀ；`C` → Aᴴ（复数共轭转置，FP32 下与 T 等价）
- `diag=UNIT`：对角元视为 1（A 中存储的对角值被忽略）；`diag=NON_UNIT`：对角元从 A 读取

与 Trsv（三角求解）不同，Trmv 是纯矩阵-向量乘：输出各元素 `y_i = Σ_j op(A)_ij · x_j` 相互独立，可完全并行，无除法、无对角非零约束。

### 特殊情况

| 输入 | 行为 |
|------|------|
| n = 0 | 空操作，直接返回成功（Strmv）；Ctrmv 约束 n ∈ [1, 8192] |
| diag = UNIT | 对角元固定为 1，A 对角存储值不参与计算 |
| incx < 0 | x 反向存储（BLAS 指针约定）：向量元素 i 位于 x[(n-1-i)·\|incx\|]（Ctrmv 仅支持 incx > 0） |
| 非 uplo 三角 | 仅被引用三角参与计算，另一三角内容不影响结果 |
| 非法参数（incx = 0、lda < n、n 越界） | 返回 ACLBLAS_STATUS_INVALID_VALUE |

## 3. 接口规范

### 算子原型

```python
cann_bench.trmv(Tensor A, Tensor x, str uplo, str trans, str diag, int incx) -> Tensor x
```

底层 C API（摘自 ops-blas README）：

```cpp
aclblasStatus_t aclblasStrmv(aclblasHandle_t handle, aclblasFillMode_t uplo,
    aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const float *A, int lda, float *x, int incx);
aclblasStatus_t aclblasCtrmv(aclblasHandle_t handle, aclblasFillMode_t uplo,
    aclblasOperation_t trans, aclblasDiagType_t diag, int64_t n,
    aclblasComplex *A, int64_t lda, aclblasComplex *x, int64_t incx);
```

### 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|----------|------|
| uplo | 输入 | ACLBLAS_UPPER(121) 上三角；ACLBLAS_LOWER(122) 下三角 |
| trans | 输入 | ACLBLAS_OP_N(111)；ACLBLAS_OP_T(112) Aᵀ；ACLBLAS_OP_C(113) Aᴴ |
| diag | 输入 | ACLBLAS_NON_UNIT(131)；ACLBLAS_UNIT(132) |
| n | 输入 | 矩阵 A 的阶数（即向量长度），n ≥ 0 |
| A | 输入 | n×lda 三角矩阵，列主序存储，仅 uplo 三角被访问 |
| lda | 输入 | 矩阵 A 存储的主维长度，lda ≥ n |
| x | 输入/输出 | 向量，输入为原始向量，输出原地覆写为结果；缓冲区长度 (n-1)·\|incx\|+1 |
| incx | 输入 | x 中连续元素之间的步长，不可为 0 |

## 4. 约束说明（摘自 ops-blas README）

- n ≥ 0（Ctrmv：n ∈ [1, 8192]）
- incx ≠ 0（Ctrmv：incx > 0）
- lda ≥ n（Ctrmv：lda > 0）
- 非法参数返回 ACLBLAS_STATUS_INVALID_VALUE

## 5. 实现参考要点（arch22 / DAV_2201 实测源码行为）

以下要点来自 `ops-blas/blas/trmv/arch22/` 的 host/kernel 源码，供实现与评审核对：

- **A 为列主序**：kernel OP_N 按 `gm_A + row + k*lda`、OP_T 按 `gm_A + k + row*lda` 取值（`strmv_kernel.cpp`，注意与 Trsv 的行主序不同）
- **incx 由 kernel 设备侧直接步进访问**（`gm_X + M0*incx*k_idx`），不同于 Trsv 的 host 侧 gather/scatter
- **strmv（FP32）**：panel 分块 tiling（M0 块），对角块按 uplo 掩码（`mask_invalid`）处理 diag 语义
- **分块流水**：MTE2 装载（A 块 + x 向量）→ Vector 矩阵-向量乘（`matrix_vector_muls_notrans`）→ 写回，双缓冲
- 输出各元素独立，可多核并行，无 Trsv 的对角线串行瓶颈

## 6. 评测注意与用例设计说明

- **原地覆写语义**：结果只写入按 incx 步长访问的位置，x 缓冲区其余位置保持输入值不变。golden 输出与 kernel 输出均须满足此语义
- **列主序解读**：golden 与 kernel 均按 `A_bl(i,j) = buf[i + j*lda]` 解读 A；对 torch 行主序 [n, lda] 张量 T 而言 `A_bl = T.t()[:n,:n]`。评审与实现请勿沿用 Trsv 的行主序约定
- **值域设计**：Trmv 无除法、输出各元素独立，不存在 Trsv 的解量级指数放大问题，值域覆盖可更自然。大 n 用例采用全正值域（A∈[1,10]、x∈[0.5,1.5]）消除累加对消，fp32 精度实测相对误差 0（阈值 0.005）；UNIT 用例对角自由
- **complex64（aclblasCtrmv）覆盖缺口说明**：当前栈（CANN 9.0.0 + torch_npu）不存在 NPU 可执行的复数矩阵-向量乘参考路径（复数 mm/tril 等基础算子缺失；linalg.solve 复数为 host 回退），故打榜用例集暂全部为 float32。proto 仍声明 complex64 能力，复数路径考核待参考实现可用后补充
- **基线口径**：torch 无三角感知的矩阵-向量乘（matmul 恒读全矩阵），e2e 参考基线为全矩阵 GEMV（torch.matmul），含约 2× 三角访存/计算量，属 torch 接口功能限制，评审可按 2× 折算理解。全部 20 条用例基线为 TTK e2e 真机实测（20/20 PASS），t_hw 按 hap-ascend-910b2-v2 skill 的 roofline 方法计算（只计三角消费字节，全部 hbm_read 瓶颈）
- **precision_thresholds**：float32=0.005、complex64=0.01（参考除法类算子的宽松阈值惯例）
