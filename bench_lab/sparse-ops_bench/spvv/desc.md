# Spvv 算子 API 描述

## 1. 算子简介

SPVV（Sparse Vector - dense Vector dot Product，稀疏向量-稠密向量点积）计算稀疏向量 x 与稠密向量 y 的点积。稀疏向量 x 以 `(indices, values)` 非零元对表示：`indices` 为严格升序且唯一的非零元索引（int32），`values` 为对应的非零元值；结果恒为单个 float32 标量。对应 ops-sparse 仓库 `sparse/spvv/arch22` 实现的 `aclsparseSpvv` 接口（参考 cuSPARSE `cusparseSpvv` 语义）。

**主要应用场景**：
- 稀疏线性代数核心内核：SpMV（稀疏矩阵-稠密向量乘）按行分解后的点积、共轭梯度法 / 双共轭梯度法等迭代求解器中的向量内积
- 稀疏检索与推荐：embedding 选中项与稠密查询向量的加权和（gather + reduce）
- 图与科学计算：稀疏图信号、有限元/有限差分离散化中的稀疏系数内积

**算子特征**：
- 难度等级：L2（Reduction）
- 输入为稀疏向量的 `(indices, values)` 非零元对与稠密向量 y，输出为单个 float32 标量
- indices shape [nnz]，values shape [nnz]，y shape [yLen]，约束 0 <= nnz <= yLen；输出 shape [1]
- 跨核归约依赖原子加（AtomicAdd）或分段规约，且 gather 的访存模式由 indices 分布决定——极稀疏时退化为标量 gather，是性能评测的关键维度

## 2. 算子定义

### 数学公式

$$
result = \sum_{i=0}^{nnz-1} values[i] \cdot y[indices[i]]
$$

其中 x 为稀疏向量，由 nnz 个非零元 `(indices[i], values[i])` 表示（索引严格升序且唯一，取值范围 [0, yLen-1]）；y 为长度 yLen 的稠密向量。等价视角：先按 `indices` 从 y 中 gather 出 nnz 个元素，与 `values` 逐元素相乘后求和。

## 3. 接口规范

### 算子原型

```python
cann_bench.spvv(indices, values, y) -> Tensor result
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| indices | Tensor | 必选 | 稀疏向量 x 的非零元索引，int32，shape [nnz]，须严格升序且唯一，取值 [0, yLen-1] |
| values | Tensor | 必选 | 稀疏向量 x 的非零元值，float16/float32，shape [nnz]，与 y 的 dtype 一致 |
| y | Tensor | 必选 | 稠密向量，float16/float32，shape [yLen] |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| result | [1] | float32 | 点积标量结果，dtype 恒为 float32（与输入 dtype 无关） |

### 数据类型

| 输入 dtype（values / y） | 输出 dtype |
|--------------------------|-----------|
| float32 | float32 |
| float16 | float32（输入 Cast 到 FP32 后乘累加） |

> indices 恒为 int32。须与 `proto.yaml` 中 inputs/outputs 的 dtype 列表完全一致。

### 规则与约束

- indices 必须严格升序且唯一，取值范围 [0, yLen-1]；nnz <= yLen。违反会导致结果未定义或越界读（对应 aclsparseSpvv 的 caller 保证项；本评测由 golden `get_input` 规整保证）
- 对应 `aclsparseSpvv` 的固定接口约束：op 仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE`，idxType 仅支持 `ACL_SPARSE_INDEX_32I`，idxBase 仅支持 `ACL_SPARSE_INDEX_BASE_ZERO`
- 累加与输出类型固定为 FP32（`computeType = ACL_FLOAT`）：FP32 输入直接 FP32 乘累加；FP16 输入先 Cast 到 FP32 再做乘法与累加（FP32 乘积），避免 half×half 的精度损失
- nnz = 0（空稀疏向量）时结果为 0
- 与同类算子的差异：`torch.dot` / `torch.vdot` 要求两个输入均为稠密向量（O(yLen) 访存）；SPVV 只读非零元（O(nnz) 访存 + gather），当 nnz << yLen 时访存量显著更小，但对 indices 的有序性与唯一性有前置要求；与 SpMV 相比，SPVV 是其按行（或按块）分解后的基本计算单元，无行指针结构

## 4. 标准 Golden 代码

```python
import torch

"""
Spvv 算子 Torch Golden 参考实现

稀疏向量-稠密向量点积（SPVV, Sparse vector - dense Vector dot Product）
公式: result = sum_{i=0}^{nnz-1} values[i] * y[indices[i]]

输出恒为 shape [1] 的 float32 标量（与输入 dtype 无关）。
"""


def spvv(
    indices: torch.Tensor,
    values: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """SPVV 同精度参考（bench, b）——忠实复现硬件累加语义

    aclsparseSpvv 的 computeType 固定 ACL_FLOAT：FP32 输入直接做 FP32 乘累加；
    FP16 输入先 Cast 到 FP32 再做乘法与累加（FP32 乘积），输出恒为 FP32。
    本函数把 values/y 升到 FP32 后做 FP32 乘累加、FP32 输出，与两条硬件路径
    的精度约定一致；对 FP32 输入 .to(float32) 为恒等变换。

    公式: result = sum_{i=0}^{nnz-1} values[i] * y[indices[i]]

    Args:
        indices: 稀疏向量非零元索引，int32，严格升序且唯一，取值 [0, yLen-1]
        values: 稀疏向量非零元值，float16/float32，shape [nnz]
        y: 稠密向量，float16/float32，shape [yLen]

    Returns:
        点积结果，float32，shape [1]
    """
    idx = indices.to(torch.int64)
    prod = values.to(torch.float32) * y.to(torch.float32)[idx]
    return torch.sum(prod, dim=0, keepdim=True)


def spvv_oracle(
    indices: torch.Tensor,
    values: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Oracle: SPVV 的数学真值 (g)。dtype-agnostic——不硬编码 .float()（=float32，
    会把 fp64 下采成 fp32），整条 gather/mul/sum 跟随输入精度；在 golden_precision=fp64_cpu
    下即真 fp64 oracle。

    plain ``spvv`` 的 .to(torch.float32) 是硬件忠实语义（computeType=ACL_FLOAT：FP16
    操作数升 FP32、FP32 累加），本身就是标准同精度参考 (b)——evaluator 的 bench 路径
    （``get_bench_function(rel_path) or golden_func``）缺 _bench 时回退 plain golden 即
    正确，故本算子默认让 plain golden 兼任 bench，只补 oracle。唯一病理是 plain 的
    .to(float32) 也把 fp64 oracle 封成 fp32，导致 |b−oracle|=0、小值域/相消严格分支
    误杀；修正 oracle 为真 fp64 后，plain-golden bench 的 FP32 累加在相消（正负抵消、
    结果接近 0）时相对 fp64 有非零误差，严格分支按 |b−g| 放松。

    公式: result = sum_{i=0}^{nnz-1} values[i] * y[indices[i]]

    Args:
        indices: 稀疏向量非零元索引，int32，严格升序且唯一，取值 [0, yLen-1]
        values: 稀疏向量非零元值，float16/float32，shape [nnz]
        y: 稠密向量，float16/float32，shape [yLen]

    Returns:
        点积结果（跟随输入精度，fp64_cpu 下为 fp64），shape [1]
    """
    idx = indices.to(torch.int64)
    prod = values * y[idx]
    return torch.sum(prod, dim=0, keepdim=True)


def get_input(
    indices: torch.Tensor,
    values: torch.Tensor,
    y: torch.Tensor,
    **kwargs,
) -> list:
    """重建 indices 为 [0, yLen) 内严格升序且唯一的随机子集，使 SPVV 良定义。

    cases.yaml 只能用单区间 value_range 表达索引，通用生成器按 randint(0, yLen-1)
    独立采样——重复与乱序几乎必然。而 aclsparseSpvv 语义要求 indices 严格升序且唯一
    （kernel 依赖有序性做二分查找 segment 边界），任何真实单算子都无法复现无序/重复
    索引下的结果。

    这里按用例声明的 nnz（= indices.shape[0]）用固定种子从 [0, yLen) 随机置换取前
    min(nnz, yLen) 项、升序排列——精确保持 nnz（去重式规整会随机缩水，密集用例
    nnz=yLen 时尤甚），且跨 eval 运行可复现。values/y 原样返回。

    kernel_eval 用输入名 + attrs 作为关键字调用本函数，并用返回值（按 golden 签名的
    Tensor 顺序）同时替换 golden 与候选的输入，故比较公平。

    Returns:
        [indices, values, y]，顺序与 spvv 签名的 Tensor 参数一致。
    """
    nnz = int(indices.shape[0])
    y_len = int(y.shape[0])
    n = min(nnz, y_len)
    g = torch.Generator().manual_seed(0)  # 固定种子：跨 eval 运行必须可复现
    picked = torch.randperm(y_len, generator=g)[:n]
    new_indices = torch.sort(picked).values.to(torch.int32)
    return [new_indices, values, y]
```

## 5. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

yLen, nnz = 10240, 1000
# 稀疏向量：[0, yLen) 内严格升序且唯一的 nnz 个索引（非零元位置）
indices = torch.randint(0, yLen, (nnz,), dtype=torch.int32).unique().sort().values.to(torch.int32)
values = torch.randn(indices.shape[0], dtype=torch.float32)
y = torch.randn(yLen, dtype=torch.float32)

result = cann_bench.spvv(indices, values, y)
# result: shape [1], float32，等价于 torch.dot(values, y[indices.to(torch.int64)])
```

### 实现要点（参考 ops-sparse arch22 kernel）

- 无 D2H 拷贝 + tiling 直传：kernel 端按 nnz 均匀分核，tiling 数据作为核函数参数传入
- 按 nnz 分核 + AtomicAdd 跨核规约：launch 前 memset 置零输出，各核部分和原子累加；每核至少 `SPVV_MIN_NNZ_PER_CORE`(512) 个元素以摊薄启动开销
- 向量化 Gather + Mul + ReduceSum，tile 内 y 片段超长时按 32B 对齐分段；indices 有序性支持二分查找 segment 边界（O(log n)）；极稀疏段回退标量 gather
- FP16 路径先 Cast 到 FP32 再 Mul（FP32 乘积），消除 half×half 在大 nnz 下的累积误差
