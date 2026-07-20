# cv_agent_matmul_bench

矩阵运算类算子 benchmark 孵化区，覆盖矩阵乘、分组矩阵乘、量化矩阵乘及融合矩阵乘等变体。所有算子 case 均符合 `docs/guide/contributing.md` 贡献规范。

每个算子子目录包含标准 5 文件：

- `proto.yaml` — 算子原型定义（schema、输入/输出、属性）
- `golden.py` — PyTorch 参考实现（仅依赖 torch）
- `desc.md` — 算子 API 文档（含标准 Golden 代码）
- `cases.yaml` — 测试用例定义（机器可读）
- `cases.csv` — 测试用例定义（与 cases.yaml 内容一致，便于人工审查）

| Operator | Case Count | Key Files |
|---|---:|---|
| `flat_quant_pertoken_int4` | 20 | `proto.yaml`, `desc.md`, `golden.py`, `cases.yaml`, `cases.csv` |
| `fused_quant_mat_mul_gelu` | 20 | `proto.yaml`, `desc.md`, `golden.py`, `cases.yaml`, `cases.csv` |
| `grouped_matmul_a8w8o16_pertoken_cv` | 20 | `proto.yaml`, `desc.md`, `golden.py`, `cases.yaml`, `cases.csv` |
| `grouped_matmul_finalize_routing_a8w8_pertoken_v1.2` | 20 | `proto.yaml`, `desc.md`, `golden.py`, `cases.yaml`, `cases.csv` |
| `grouped_matmul_swiglu_quant_a8w8_tiling0` | 20 | `proto.yaml`, `desc.md`, `golden.py`, `cases.yaml`, `cases.csv` |
| `quant_batch_matmul_inplace_add_mx_dynamic` | 20 | `proto.yaml`, `desc.md`, `golden.py`, `cases.yaml`, `cases.csv` |
| `quant_batch_matmul_v3_basic_epilogue` | 20 | `proto.yaml`, `desc.md`, `golden.py`, `cases.yaml`, `cases.csv` |
| `quant_batch_matmul_v4_gb_dynamic_pergroup` | 20 | `proto.yaml`, `desc.md`, `golden.py`, `cases.yaml`, `cases.csv` |
| `quant_grouped_matmul_dequant_dynamic_pertoken` | 20 | `proto.yaml`, `desc.md`, `golden.py`, `cases.yaml`, `cases.csv` |
| `quant_grouped_matmul_inplace_add_tc_perchannel` | 20 | `proto.yaml`, `desc.md`, `golden.py`, `cases.yaml`, `cases.csv` |
| `quant_matmul_reduce_sum_weight_nz` | 20 | `proto.yaml`, `desc.md`, `golden.py`, `cases.yaml`, `cases.csv` |
| `rotate_quant` | 20 | `proto.yaml`, `desc.md`, `golden.py`, `cases.yaml`, `cases.csv` |
| `sparse4to2quant_matmul_dequant_weight_nz` | 23 | `proto.yaml`, `desc.md`, `golden.py`, `cases.yaml`, `cases.csv` |
| `transpose_quant_batch_mat_mul_kc_debug` | 20 | `proto.yaml`, `desc.md`, `golden.py`, `cases.yaml`, `cases.csv` |
| `weight_quant_batch_matmul_v2_antiquant` | 20 | `proto.yaml`, `desc.md`, `golden.py`, `cases.yaml`, `cases.csv` |
