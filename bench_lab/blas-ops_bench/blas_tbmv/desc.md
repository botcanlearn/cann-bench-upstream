# Tbmv 算子 API 描述

## 1. 算子简介

Tbmv（Triangular Band Matrix-Vector Multiplication）实现三角**带状**矩阵与向量的乘法 `y = op(A) * x`，是 BLAS 核心例程之一（对应 cuBLAS 的 `tbmv`）。相比稠密三角乘（Trmv），A 仅存储半带宽 k 内的对角线，访存量从 n² 降至 (k+1)·n。ops-blas 提供单精度实数接口 `aclblasStbmv`（及早期接口 `aclblasStbmv_legacy`）。

**主要应用场景**：
- 带状线性系统的迭代求解子步（如三对角/五对角差分方程）
- 样条插值、带状稀疏结构的科学与工程计算

**算子特征**：
- 难度等级：L2（Contraction）—— 带状访存模式 + 属性组合；输出各元素相互独立，无串行依赖
- **arch22 语义为 x/y 分离输出**（README 数学表达式："y = A * x（arch22，输入输出分离）"；arch35 为原地覆盖 x = op(A)*x）
- 仅 FP32 接口（README 未定义复数 Tbmv 接口）

## 2. 算子定义

### 数学公式

$$
y = op(A) \cdot x, \quad A \in \mathbb{R}^{n \times n} \text{ 三角带状（半带宽 } k\text{）}
$$

- `uplo=UPPER`：仅引用 A 的第 0..k 条上对角线；`uplo=LOWER`：仅引用第 0..k 条下对角线
- `trans` 决定 op(A)：`N` → A；`T` → Aᵀ；`C` → Aᴴ（实数下与 T 等价）
- `diag=UNIT`：对角元视为 1；`diag=NON_UNIT`：对角元从 A 读取
- `k=0` 时退化为纯对角乘；`k=n-1` 时退化为满三角 Trmv

### 特殊情况

| 输入 | 行为 |
|------|------|
| n ≤ 0 | 空操作，直接返回成功（legacy host 校验：n ≤ 0 return SUCCESS） |
| diag = UNIT | 对角元固定为 1（乘法因子，存储对角值不影响合法性） |
| incx < 0 | x 反向存储（BLAS 指针约定），kernel 经 XPhysicalPos 支持正负步长 |
| k = 0 | 仅对角线参与计算 |
| 非法参数（uplo/trans/diag 越界、incx = 0） | 返回 ACLBLAS_STATUS_INVALID_VALUE |

## 3. 接口规范

### 算子原型

```python
cann_bench.tbmv(Tensor A, Tensor x, str uplo, str trans, str diag, int k, int incx) -> Tensor y
```

底层 C API（摘自 ops-blas README）：

```cpp
aclblasStatus_t aclblasStbmv(aclblasHandle_t handle, aclblasFillMode_t uplo,
    aclblasOperation_t trans, aclblasDiagType_t diag, int n, int k,
    const float *A, int lda, float *x, int incx);
// arch22 实现为 legacy 签名（x/y 分离）:
aclblasStatus_t aclblasStbmv_legacy(aclblasHandle_t handle, aclblasFillMode uplo,
    aclblasOperation trans, aclblasDiagType diag, const float* a, int64_t lda,
    const float* x, float* y, int64_t n, int64_t k, int64_t incx);
```

### 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|----------|------|
| uplo | 输入 | ACLBLAS_UPPER(121) 上三角；ACLBLAS_LOWER(122) 下三角 |
| trans | 输入 | ACLBLAS_OP_N(111)；ACLBLAS_OP_T(112) Aᵀ；ACLBLAS_OP_C(113) Aᴴ |
| diag | 输入 | ACLBLAS_NON_UNIT(131)；ACLBLAS_UNIT(132) |
| n | 输入 | 矩阵 A 的阶数，n ≥ 0 |
| k | 输入 | 半带宽，k ≥ 0 |
| A | 输入 | lda×n 带状打包缓冲，列主序带状存储（见下） |
| lda | 输入 | 主维长度，lda ≥ k+1 |
| x | 输入 | 向量，长度 (n-1)·\|incx\|+1，按 incx 步长访问 |
| y | 输出 | 结果向量（arch22 分离输出） |
| incx | 输入 | x 的存储增量，不可为 0 |

注：`n`、`lda` 为底层 C API 形参，cann_bench python 接口（proto schema）不单独暴露——n 由 A.shape[1] 与 x 长度隐含，lda 由 A.shape[0] 隐含（A 为 [lda, n] 带状打包缓冲，与 Trsv/Trmv 的 [n, lda] 布局相反）。cases.yaml/csv 的 attrs 键名仅与 proto 声明的 uplo/trans/diag/k/incx 对应（contributing.md §4.2）。

### 带状打包布局（README 调用示例解码验证）

示例 n=4, k=1, lda=2，`hA = {4,1, 5,2, 6,3, 7,0}` 对应逻辑矩阵 A(0,0)=4, A(1,0)=1, A(1,1)=5, A(2,1)=2, A(2,2)=6, A(3,2)=3, A(3,3)=7 —— 即 **BLAS 标准列主序带状打包**：

- 下三角：`A(i,j) = buf[(i-j) + j*lda]`（打包第 b=i-j 行存放第 b 条下对角线）
- 上三角：`A(i,j) = buf[(k+i-j) + j*lda]`（打包第 k-d 行存放第 d 条上对角线，BLAS 标准约定）

## 4. 约束说明（摘自 ops-blas README）

- n ≥ 0（legacy host：n ≤ 0 直接返回成功）
- k ≥ 0
- lda ≥ k+1
- incx ≠ 0
- 非法参数返回 ACLBLAS_STATUS_INVALID_VALUE

## 5. 实现参考要点（arch22 / DAV_2201 实测源码行为）

- **x/y 分离输出**：arch22 host 实现 `aclblasStbmv_legacy(a, lda, x, y, n, k, incx)`，y 为独立输出指针（`stbmv_host.cpp`）；README 数学表达式标注 arch22 与 arch35（原地）的差异
- **带状打包**：打包缓冲按 (k+1) 条对角线组织，UNIT 语义通过将对角打包行覆写为 1 实现（kernel `diag == ACLBLAS_UNIT && bandIdx == 0` 分支）
- **⚠️ arch22 打包约定与本任务 golden 不同**：arch22 kernel 索引为 `a[bandIdx*lda + col]`（对角主序，lda 为对角线间步长，官方测试用 `(k+1)*lda` 缓冲且隐含 lda ≥ n）；本任务 golden 与 README「维度 lda×n、lda ≥ k+1」、arch35 fallback（`buf[(k+i-j)+j*lda]`）、官方 cblas ColMajor golden 一致，为 BLAS 列主序带状打包。实现本算子 kernel 时请按 golden/BLAS 公式解码，勿照抄 arch22 索引（两者互为转置关系）
- **多核 tiling**：taskCount = k+1（每条对角线一个任务），useCoreNum = min(taskCount, 8)（incx==1 时）；incx≠1 时单核
- **x 步进访问**：kernel 经 XPhysicalPos 支持正负 incx；y 按对角线逐条累加（跨对角线求和）

## 6. 评测注意与用例设计说明

- **x/y 分离语义**：golden 返回长度 n 的稠密结果向量，不覆写 x；y 的 incx 仅为存储布局，数学结果与 incx 无关
- **k 覆盖设计**：k=0（纯对角）、k=1、中带宽、k=n-1（满三角退化）均有用例；性能锚点扫描 k/n 比例（n/8 与 n-1 两极）
- **值域设计**：带状乘无除法、输出各元素独立，无解爆炸问题。大 n 用例采用全正值域（A∈[1,10]、x∈[0.5,1.5]）消除累加对消；UNIT 用例对角自由
- **基线口径**：torch 无带状 matvec。基线由算子拼接得到（cann-bench 惯例："由于 torch 接口功能限制，部分基线实现由算子拼接得到"）：滑动窗口（as_strided）+ 逐元素乘 + 归约求和，数学上精确等价于带状乘，工作量 O((k+1)·n)，访存约 3× 带状理想值；以 warmup=5、repeat=20 的同步墙钟均值采集
- **t_hw**：按 hap-ascend-910b2-v2 skill 的 roofline 方法计算，只计被消费的带内字节（Σ_{b=0}^{k}(n-b)·4），全部用例 hbm_read 瓶颈
- **precision_thresholds**：float32=0.005（参考除法类算子的宽松阈值惯例）
