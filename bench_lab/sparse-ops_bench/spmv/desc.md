# Spmv 算子 API 描述

## 1. 算子简介

SpMV（Sparse Matrix - Dense Vector Multiplication，稀疏矩阵-稠密向量乘）计算 CSR 格式稀疏矩阵 A 与稠密向量 x 的乘法，并按 `y_out = alpha * op(A) * x + beta * y` 与输入向量 y 组合（beta=0 时为纯 SpMV，alpha=beta=1 的转置模式等价于 Aᵀx）。对应 ops-sparse 仓库 `sparse/spmv/arch22` 实现的 `aclsparseSpMV` 接口（参考 cuSPARSE `cusparseSpMV` 语义），支持非转置与转置两种模式。

**主要应用场景**：
- 稀疏线性代数核心内核：迭代求解器（共轭梯度法、GMRES 等）每步迭代的矩阵-向量乘、特征值计算（Lanczos/Arnoldi）
- 图神经网络与图计算：稀疏邻接矩阵与节点特征的乘法（消息传递的加权汇聚）
- 推荐系统与科学计算：稀疏投影、稀疏正则项梯度、PageRank 幂迭代

**算子特征**：
- 难度等级：L3（Contraction）
- 输入为 CSR 三数组 `(csr_row_ptr, csr_col_ind, csr_val)` 与稠密向量 x、y，输出为与 y 同 shape 的稠密向量
- 非转置 x [N]、y [M]；转置 x [M]、y [N]；csr_row_ptr [M+1]，csr_col_ind/csr_val [nnzA]
- 变长行分解 + 跨核归约（转置模式经 AtomicAdd 散射累加），行分布不均衡与极稀疏/高密度两端是性能评测的关键维度

## 2. 算子定义

### 数学公式

$$
y\_out = \alpha \cdot op(A) \cdot x + \beta \cdot y
$$

其中 A 为 M×N CSR 稀疏矩阵，op(A) 为 A（非转置）或 Aᵀ（转置）；逐元素形式：

$$
\text{非转置: } y\_out_i = \alpha \sum_{j \in row(i)} A_{ij} x_j + \beta y_i, \qquad
\text{转置: } y\_out_j = \alpha \sum_{i: A_{ij} \neq 0} A_{ij} x_i + \beta y_j
$$

CSR 格式由三个数组表示：`csr_row_ptr[i]`（第 i 行第一个非零元在值数组中的偏移，长度 M+1，单调不减且末位为 nnzA）、`csr_col_ind`（非零元列索引，行内严格升序且唯一）、`csr_val`（非零元值）。

## 3. 接口规范

### 算子原型

```python
cann_bench.spmv(csr_row_ptr, csr_col_ind, csr_val, x, y, alpha=1.0, beta=0.0, transpose=False) -> Tensor y_out
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| csr_row_ptr | Tensor | 必选 | CSR 行偏移数组，int32，shape [M+1]，单调不减，csr_row_ptr[M] = nnzA |
| csr_col_ind | Tensor | 必选 | CSR 列索引数组，int32，shape [nnzA]，行内严格升序且唯一，取值 [0, N-1] |
| csr_val | Tensor | 必选 | CSR 非零元值，float16/float32/bfloat16，shape [nnzA]，与 x 的 dtype 一致 |
| x | Tensor | 必选 | 输入稠密向量，非转置 [N]、转置 [M]，与 csr_val 的 dtype 一致 |
| y | Tensor | 必选 | 输入稠密向量（beta*y 的被乘向量），非转置 [M]、转置 [N] |
| alpha | float | 1.0 | 标量系数，乘在 op(A)*x 上 |
| beta | float | 0.0 | 标量系数，乘在输入 y 上 |
| transpose | bool | false | true=op(A)=Aᵀ（转置模式），false=op(A)=A |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y_out | 与 y 相同 | 与输入 y 相同 | 结果向量；FP16/BF16 输入按 FP32 累加后写回 y dtype |

### 数据类型

| 输入 dtype（csr_val / x） | 输出 dtype（y / y_out） |
|---------------------------|------------------------|
| float32 | float32 |
| float16 | float16 或 float32 |
| bfloat16 | bfloat16 或 float32 |

> csr_row_ptr / csr_col_ind 恒为 int32。须与 `proto.yaml` 中 inputs/outputs 的 dtype 列表完全一致。

### 规则与约束

- CSR 结构须良定义：csr_row_ptr 单调不减且 csr_row_ptr[M] = nnzA；行内列索引严格升序且唯一，取值 [0, N-1]（canonical CSR，本评测由 golden `get_input` 规整保证；对应 aclsparseSpMV 的 caller 保证项）
- 维度匹配：非转置 x 长度 ≥ N、y 长度 ≥ M；转置 x 长度 ≥ M、y 长度 ≥ N；nnzA ≤ M*N
- 对应 `aclsparseSpMV` 的固定接口约束：格式仅支持 CSR，索引类型仅支持 `ACL_SPARSE_INDEX_32I`，idxBase 仅支持 `ACL_SPARSE_INDEX_BASE_ZERO`，alg 仅支持 `ACL_SPARSE_SPMV_ALG_DEFAULT`
- 累加精度：computeType 固定 `ACL_FLOAT`——FP16/BF16 输入 Cast 到 FP32 后做乘法，ReduceSum 固定 float 精度，alpha/beta 以 FP32 参与运算，结果写回 y 的 dtype（原接口的 int32 精确整数计算路径 `computeType=ACL_INT32` 不在本任务覆盖范围）
- nnzA = 0（空矩阵）时 y_out = beta * y
- 单行最大非零元数受 kernel UB 容量限制（`maxTileLength ≈ (UB_SIZE - 4KB) / 20B`，float 计算约为 1 万量级），用例设计已避开
- 与同类算子的差异：`torch.mv` 要求稠密矩阵（O(M*N) 访存）；SpMV 只读非零元（O(nnzA) 访存 + gather/scatter），但需处理变长行分解与（转置模式下的）散射累加；与 SpMM（矩阵-矩阵）相比无第二稠密矩阵维度；与 SpVV（向量-向量点积）相比，SpVV 是 SpMV 按行分解后的单行计算单元

## 4. 标准 Golden 代码

```python
import torch

"""
Spmv 算子 Torch Golden 参考实现

稀疏矩阵-稠密向量乘（SpMV, Sparse Matrix - Dense Vector Multiplication，CSR 格式）
公式: y_out = alpha * op(A) * x + beta * y
      非转置 op(A)=A:   y_out[i] = alpha * sum_{j in row(i)} A[i,j] * x[j] + beta * y[i]
      转置   op(A)=A^T: y_out[j] = alpha * sum_{i: A[i,j]!=0} A[i,j] * x[i] + beta * y[j]

输出 dtype 与输入 y 一致（FP16/BF16 输入按 computeType=ACL_FLOAT 以 FP32 乘累加后写回）。
"""


def _expand_rows(row_ptr: torch.Tensor, nnz: int) -> torch.Tensor:
    """求每个非零元（按 CSR 线性位置 0..nnz-1）所在的行号。

    row_ptr[1:] 为各行终点，非零元 k 属于第 r 行当且仅当 row_ptr[r] <= k < row_ptr[r+1]，
    即 r = #{ends <= k} = searchsorted(ends, k, right=True)。空行（终点相等）自然被跳过。
    """
    return torch.searchsorted(row_ptr[1:].contiguous(), torch.arange(nnz), right=True)


def spmv(
    csr_row_ptr: torch.Tensor,
    csr_col_ind: torch.Tensor,
    csr_val: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.0,
    transpose: bool = False,
) -> torch.Tensor:
    """SpMV 同精度参考（bench, b）——忠实复现硬件累加语义

    aclsparseSpMV 的 computeType 固定 ACL_FLOAT：FP32 输入直接 FP32 乘累加；FP16/BF16
    输入先 Cast 到 FP32 再做乘法，ReduceSum 固定 float 精度，alpha/beta 以 FP32 参与运算，
    结果写回 y 的 dtype。本函数把 csr_val/x/y 升到 FP32 后做 FP32 乘累加（index_add 的
    FP32 累加对应核内 ReduceSum/AtomicAdd 的 FP32 累加器），最后 cast 回 y.dtype，与硬件
    路径的精度约定一致；对 FP32 输入 .to(float32) 为恒等变换。

    公式: y_out = alpha * op(A) * x + beta * y

    Args:
        csr_row_ptr: CSR 行偏移数组，int32，shape [M+1]，单调不减
        csr_col_ind: CSR 列索引数组，int32，shape [nnzA]，行内严格升序且唯一
        csr_val: CSR 非零元值，float16/float32/bfloat16，shape [nnzA]
        x: 输入稠密向量，float16/float32/bfloat16（非转置 [N]，转置 [M]）
        y: 输入稠密向量，float16/float32/bfloat16（非转置 [M]，转置 [N]）
        alpha: 标量系数，乘在 op(A)*x 上
        beta: 标量系数，乘在输入 y 上
        transpose: false=op(A)=A，true=op(A)=Aᵀ

    Returns:
        结果向量 y_out，dtype 与 y 一致，shape 与 y 相同
    """
    row_ptr = csr_row_ptr.to(torch.int64)
    col_idx = csr_col_ind.to(torch.int64)
    vals = csr_val.to(torch.float32)
    nnz = int(csr_val.shape[0])
    acc = torch.zeros(int(y.shape[0]), dtype=torch.float32)
    if transpose:
        # 按行号 gather x、按列号累加：y_out[j] += A[i,j] * x[i]
        prod = vals * x.to(torch.float32)[_expand_rows(row_ptr, nnz)]
        acc.index_add_(0, col_idx, prod)
    else:
        # 按列号 gather x、按行号累加：y_out[i] += A[i,j] * x[j]
        prod = vals * x.to(torch.float32)[col_idx]
        acc.index_add_(0, _expand_rows(row_ptr, nnz), prod)
    out = alpha * acc + beta * y.to(torch.float32)
    return out.to(y.dtype)


def spmv_oracle(
    csr_row_ptr: torch.Tensor,
    csr_col_ind: torch.Tensor,
    csr_val: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.0,
    transpose: bool = False,
) -> torch.Tensor:
    """Oracle: SpMV 的数学真值 (g)。dtype-agnostic——不硬编码 .float()（=float32，
    会把 fp64 下采成 fp32），整条 gather/mul/index_add 跟随输入精度；在
    golden_precision=fp64_cpu 下即真 fp64 oracle。

    plain ``spmv`` 的 .to(torch.float32) 是硬件忠实语义（computeType=ACL_FLOAT：FP16/BF16
    操作数升 FP32、FP32 累加、写回 y dtype），本身就是标准同精度参考 (b)——evaluator 的
    bench 路径（``get_bench_function(rel_path) or golden_func``）缺 _bench 时回退
    plain golden 即正确，故本算子默认让 plain golden 兼任 bench，只补 oracle。唯一病理是
    plain 的 .to(float32) 也把 fp64 oracle 封成 fp32，导致 |b−oracle|=0、小值域/相消严格
    分支误杀；修正 oracle 为真 fp64 后，plain-golden bench 的 FP32 累加在相消行（正负
    抵消、行和接近 0）相对 fp64 有非零误差，严格分支按 |b−g| 放松。

    公式: y_out = alpha * op(A) * x + beta * y

    Args:
        csr_row_ptr: CSR 行偏移数组，int32，shape [M+1]，单调不减
        csr_col_ind: CSR 列索引数组，int32，shape [nnzA]，行内严格升序且唯一
        csr_val: CSR 非零元值，float16/float32/bfloat16，shape [nnzA]
        x: 输入稠密向量，float16/float32/bfloat16（非转置 [N]，转置 [M]）
        y: 输入稠密向量，float16/float32/bfloat16（非转置 [M]，转置 [N]）
        alpha: 标量系数，乘在 op(A)*x 上
        beta: 标量系数，乘在输入 y 上
        transpose: false=op(A)=A，true=op(A)=Aᵀ

    Returns:
        结果向量（跟随输入精度，fp64_cpu 下为 fp64），shape 与 y 相同
    """
    row_ptr = csr_row_ptr.to(torch.int64)
    col_idx = csr_col_ind.to(torch.int64)
    nnz = int(csr_val.shape[0])
    acc = torch.zeros(int(y.shape[0]), dtype=csr_val.dtype)
    if transpose:
        prod = csr_val * x[_expand_rows(row_ptr, nnz)]
        acc.index_add_(0, col_idx, prod)
    else:
        prod = csr_val * x[col_idx]
        acc.index_add_(0, _expand_rows(row_ptr, nnz), prod)
    return alpha * acc + beta * y


def get_input(
    csr_row_ptr: torch.Tensor,
    csr_col_ind: torch.Tensor,
    csr_val: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    transpose: bool = False,
    **kwargs,
) -> list:
    """重建 CSR 结构（row_ptr 单调不减、行内列索引严格升序且唯一），使 SpMV 良定义。

    cases.yaml 只能用单区间 value_range 表达索引，通用生成器按 randint 独立采样——
    row_ptr 乱序且不单调、col_ind 行内重复/乱序几乎必然。而 CSR 的良定义性要求
    row_ptr 单调且 csr_row_ptr[M]=nnzA、行内列索引严格升序唯一（canonical CSR，
    kernel 的行分解/二分均依赖此结构），任何真实单算子都无法复现乱序结构下的行为。

    规整策略（保持 nnzA、M、N 与用例声明一致，csr_val/x/y 原样保留）：
    1. nnzA 个非零元用固定种子均匀随机落入 M 行（多项式分布，允许空行），
       cumsum 得 row_ptr——行分布随机、贴近真实稀疏矩阵；
    2. 行内列索引：每个非零元采样 u∈[0,1)，行内按 u 排序后量化到 [0, N-maxc)
       （maxc=最大行长），再加行内秩——量化值行内非降，加严格递增的秩后必严格
       递增，且落在 [0, N-1] 内、行内无重复，分布近似均匀；
    3. 固定种子保证跨 eval 运行可复现（精度二次验证换新值时结构不变）。

    kernel_eval 用输入名 + attrs 作为关键字调用本函数，并用返回值（按 golden 签名的
    Tensor 顺序）同时替换 golden 与候选的输入，故比较公平。

    Returns:
        [csr_row_ptr, csr_col_ind, csr_val, x, y]，顺序与 spmv 签名的 Tensor 参数一致。
    """
    nnz = int(csr_val.shape[0])
    if transpose:
        m_rows, n_cols = int(x.shape[0]), int(y.shape[0])
    else:
        m_rows, n_cols = int(y.shape[0]), int(x.shape[0])
    g = torch.Generator().manual_seed(0)  # 固定种子：跨 eval 运行必须可复现
    if nnz == 0:
        row_ptr = torch.zeros(m_rows + 1, dtype=torch.int32)
        col_ind = torch.empty(0, dtype=torch.int32)
        return [row_ptr, col_ind, csr_val, x, y]

    # 1) 行分布：nnzA 个非零元均匀随机落入 M 行（多项式分布，允许空行）
    row_ids = torch.randint(0, m_rows, (nnz,), generator=g)
    counts = torch.bincount(row_ids, minlength=m_rows)
    maxc = int(counts.max())
    assert maxc <= n_cols, (
        f"最大行长 {maxc} 超过列数 {n_cols}，行内唯一列索引不存在；"
        "请调整用例的 M/N/nnzA（nnzA/M 应显著小于 N）")

    # 2) 行内严格升序唯一列索引：按 (row, u) 双稳定排序后量化 + 行内秩
    u = torch.rand(nnz, generator=g)
    order = torch.argsort(u, stable=True)              # 次关键字 u 升序
    order = order[torch.argsort(row_ids[order], stable=True)]  # 主关键字行号稳定排序
    u_o, rows_o = u[order], row_ids[order]
    starts = counts.cumsum(0) - counts                 # 每行起始位置（排他前缀和）
    rank = torch.arange(nnz) - starts[rows_o]          # 行内秩 0..c_r-1
    col_ind = ((u_o * (n_cols - maxc)).to(torch.int64) + rank).to(torch.int32)

    # 3) row_ptr = [0, cumsum(counts)]，单调不减，末位 = nnzA
    row_ptr = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)]).to(torch.int32)
    return [row_ptr, col_ind, csr_val, x, y]
```

## 5. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

M, N, nnzA = 1024, 2048, 65536
# CSR 结构：row_ptr 单调不减且末位为 nnzA，行内列索引严格升序唯一（canonical CSR）
csr_row_ptr = torch.zeros(M + 1, dtype=torch.int32)
csr_col_ind = torch.randint(0, N, (nnzA,), dtype=torch.int32).sort().values
csr_row_ptr[1:] = torch.linspace(0, nnzA, M + 1).to(torch.int32)  # 示意：实际需按行切分
csr_val = torch.randn(nnzA, dtype=torch.float32)
x = torch.randn(N, dtype=torch.float32)
y = torch.randn(M, dtype=torch.float32)

y_out = cann_bench.spmv(csr_row_ptr, csr_col_ind, csr_val, x, y,
                        alpha=1.0, beta=0.0, transpose=False)
# y_out: shape [M]，dtype 与 y 相同
# 等价稠密运算：A = 稀疏化(csr_row_ptr, csr_col_ind, csr_val); y_out = A @ x
```

### 实现要点（参考 ops-sparse arch22 kernel）

- 类型独立编译：7 个 dtype 组合（fp32/fp16/bf16 × fp32 输出、int32）各自实例化模板 kernel，`trans` 标量控制非转置/转置分发
- 行并行策略：每个 AI Core 处理若干整行，核间均分 workload；转置模式结果经原子累加写入避免核间踩踏
- CopyIn → Compute → CopyOut 三级流水，行内乘累加经 UB 缓冲复用；混合精度经矢量 Cast 完成，ReduceSum 固定 float 精度
- Host 侧 D2H 读 row_ptr 计算最大行长做 UB 容量检查（单行超限返回 `ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES`）
