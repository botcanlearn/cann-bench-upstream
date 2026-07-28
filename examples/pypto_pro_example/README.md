# PyPTO-Pro CANN Example

基于 PyPTO-Pro (Python-to-NPU compiler, Pro 版) 的 NPU 算子示例，打包为 `cann_bench` Python 包供 CANN Bench 评测。本示例由 PyPTO-Pro 算子开发工作流（pypto-pro-op-orchestrator）生成，作为后续 PyPTO-Pro 算子产物提交给 CANN Bench 的样板。

本示例包含一个算子 `rms_norm`，隔离在 `cann_bench/rms_norm/` 子包中。该结构同时演示了多算子提交的目录约定（见下文）。

## 目录结构

```
pypto_pro_example/
├── cann_bench/                         # Python 包
│   ├── __init__.py                     # 顶层 thin forwarder（聚合所有算子入口）
│   └── rms_norm/                       # 算子子包：RmsNorm (L2)
│       ├── __init__.py                 # from .dispatcher import rms_norm
│       ├── dispatcher.py               # 多 class (ndim,dtype) 分发器
│       └── c1/ ~ c7/
│           └── test_rms_norm.py        # @pl.jit kernel + host wrapper
├── build.sh                            # 构建脚本
├── setup.py                            # 打包配置
└── README.md
```

## 多算子隔离约定

每个算子独占一个 `cann_bench/<op>/` 子包，内含：
- `__init__.py`：从 dispatcher 导出入口函数
- `dispatcher.py`：多 class 分发器（按 `(ndim, dtype)` 签名路由）
- `c{N}/test_<op>.py`：每个 class 的 `@pl.jit` kernel + host wrapper

顶层 `cann_bench/__init__.py` 为每个算子写一个 thin forwarder，评测框架通过 `dir(cann_bench)` 扫描到所有算子接口。新增算子时只需：
1. 新建 `cann_bench/<op>/` 子包（复制上述结构）
2. 顶层 `__init__.py` 追加一个 forwarder
3. `setup.py` 的 `packages` 和 `package_data` 追加该子包

## 算子说明

### RmsNorm (L2 / Normalization)

$$y = \frac{x}{\sqrt{\mathrm{mean}(x^2) + \epsilon}} \cdot \gamma$$

- 输入：x（fp16/fp32/bf16，任意前导维度+末维 D）、gamma（1D [D]）
- 输出：y（同输入 shape/dtype）
- attrs：epsilon（float，默认 1e-6）
- 7 个 class (c1~c7) 覆盖不同 (ndim, dtype) 签名

## 构建方法

在 cann-bench 仓根目录下执行：

```bash
bash examples/pypto_pro_example/build.sh           # 构建 wheel 包到 examples/pypto_pro_example/dist/
```

## 评测方法

### 前置条件

1. PyPTO 已安装
2. NPU 设备可用

> **工作目录**：以下所有命令（`PYTHONPATH=src`、`--task-dir tasks/...`、`--source-dir examples/...` 均为相对路径）必须在 **cann-bench 仓根目录**下执行，即先 `cd <path-to>/cann-bench`。

### 评测算子

```bash
# 评测 RmsNorm (L2)
PYTHONPATH=src python -m kernel_eval.cli eval \
  --bench-name cann \
  --task-dir tasks/level2/rms_norm \
  --source-dir examples/pypto_pro_example \
  --device-id 0 \
  --reports-dir "$PWD/reports"
```

> **`--reports-dir` 必须用绝对路径**：PyPTO-Pro 算子在每个用例的独立子进程中执行，子进程会 `chdir` 到临时目录隔离 JIT 编译。若 `--reports-dir` 为相对路径（如默认的 `reports`），profiler 产出的 `kernel_details.csv` 等性能数据会落到 chdir 后的临时目录，父进程在项目根下找不到，导致耗时/加速比显示为 `N/A`、性能得分为 0。用 `"$PWD/reports"` 或绝对路径可避免此问题。

### 报告输出位置

报告保存到 `--reports-dir` 指定的目录（上例为 `<repo_root>/reports/`），包含三种格式：
- `{eval_code}.json` — 完整结构化数据（含 per-case elapsed_us、accuracy、kernel details 指标）
- `{eval_code}.md`   — Markdown 摘要（概览表 + 每算子详情表）
- `{eval_code}.html` — 独立可视化报告

### 评测流程

```
python -m kernel_eval.cli eval --source-dir examples/pypto_pro_example
  │
  ├─ build.sh → cann_bench-1.0.0-py3-none-any.whl
  ├─ pip install --no-deps cann_bench-1.0.0-py3-none-any.whl
  ├─ import cann_bench → 扫描接口: rms_norm
  ├─ 匹配 tasks/levelN/<op> 中的算子定义
  │
  └─ 逐用例评测:
      ├─ 加载 cases.yaml 用例
      ├─ 生成输入数据
      ├─ 执行 golden 参考（CPU fp64）
      ├─ 执行 AI 算子 (NPU + Profiler)
      ├─ 精度对比（MERE/MARE）
      ├─ 性能：默认 kernel_details 策略
      └─ 性能评分（HAP）
```

## 调用链

```
cann_bench/__init__.py
  → cann_bench.<op>  (子包)
    → <op>/__init__.py
      → <op>/dispatcher.py            # 按 (ndim, dtype) 签名匹配 c{N}
        → c{N}/test_<op>.py           # 实际 kernel (<op>_wrapper)
```

## PyPTO-Pro Kernel 结构

每个 `c{N}/test_<op>.py` 包含三部分：

| 部分 | 职责 | 示例 |
|------|------|------|
| **vector_function** | `@pl.vector_function` 装饰的 vector 计算函数 | `rms_norm_rows_vf(in_tile, out_tile, gamma_tile, ...)` |
| **Kernel 定义** | `@pl.jit(auto_mutex=True)` 装饰的 JIT kernel | `rms_norm_kernel(x, gamma, epsilon, y)` |
| **Host wrapper** | 折叠前导维、分配输出、调用 JIT | `rms_norm_wrapper(x, gamma, epsilon=1e-6)` |

JIT kernel 通过 `pl.TileType` + `pl.make_tile_group` 构造 tile 分组，用 SPMD 跨步循环（`pl.get_block_num` / `pl.get_block_idx`）遍历 M 维 tile，`pl.load` / `pl.store` 完成 GM↔UB 搬运。

## 多 Class 分发机制

各算子 `dispatcher.py` 中的 `_CLASSES` 表将输入 tensor 的 `(ndim, dtype)` 签名路由到对应 class 的 kernel。dispatcher 通过 `importlib` 按文件路径懒加载对应 class 的 `test_<op>.py`，入口函数查找顺序为 `<op>` → `<op>_wrapper`。

### RmsNorm

| Class | x ndim | x dtype  | gamma ndim | gamma dtype |
|-------|--------|----------|------------|-------------|
| c1    | 3D     | float16  | 1D         | float16     |
| c2    | 3D     | float32  | 1D         | float32     |
| c3    | 3D     | bfloat16 | 1D         | bfloat16    |
| c4    | 2D     | bfloat16 | 1D         | bfloat16    |
| c5    | 4D     | float16  | 1D         | float16     |
| c6    | 4D     | float32  | 1D         | float32     |
| c7    | 5D     | float32  | 1D         | float32     |

> 各 class 的 tile 参数（MAX_N/TILE_ROWS）、地址布局、dtype 转换策略、计算顺序均按 (ndim, dtype) 特化，不可合并为单一实现。