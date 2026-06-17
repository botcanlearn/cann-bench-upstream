# MatmulAllReduce Operator Description

## 1. Overview

MatmulAllReduce is an MC2 fused communication-compute operator. It computes the local `x1 @ x2`, optionally adds `bias`, and then performs an HCCL all-reduce with `reduce_op=sum`.

This benchmark task contains 20 real non-quantized cases selected from `mc2_test/excel/aclnnMatmulAllReduce.xlsx`. The selection follows these constraints:

- Atlas A2 8-card cases, `world_size=8`.
- Single-op mode, `graph_type=0`.
- Non-quantized float16 / bfloat16 inputs.
- Priority is given to the `level0` sheet. The final set uses 15 large and diverse `level0` cases and 5 large `level1` fp16 cases to avoid a bfloat16-only set.
- Shapes cover multiple model families and dimensions, including x1_65b, x1_175b, xiaoyi, z1_200b, and ks_65b.

The source row is recorded in each case under `attrs.source_sheet` and `attrs.source_excel_row`.

## 2. Definition

```text
local = x1 @ x2 (+ bias)
y = AllReduce(local, op=sum)
```

Input and output semantics:

- `x1`: rank-local matmul input, shape `[..., M, K]`.
- `x2`: matmul weight, shape `[K, N]`; the selected cases currently use `is_trans_b=false`.
- `bias`: optional bias, shape `[N]`.
- `y`: all-reduced matmul result, same shape as the local matmul output.

## 3. Interface

```python
cann_bench.matmul_all_reduce(
    x1,
    x2,
    hcomm_info: str,
    world_size: int,
    reduce_op: str = "sum",
    bias=None,
    is_trans_b: bool = False,
) -> Tensor
```

Cases set `attrs.mc2_distributed: true`, so cann-bench routes them through the MC2 distributed runner. The runner initializes one HCCL process per rank and passes the communicator name to the candidate implementation.

## 4. Accuracy

The float16 and bfloat16 thresholds are both `0.05`. The task-local `golden.py` uses `torch.matmul` plus `torch.distributed.all_reduce` in each rank process to produce the reference output.
