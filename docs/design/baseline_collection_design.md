# Baseline 性能采集脚本设计

**版本: v3-draft | 日期: 2026-06-06 | 更新: refs 迁移策略 + golden.py vs refs 区分**

---

## 1. 背景

当前 baseline 性能数据 (`baseline_perf_us` / `t_hw_us`) 的采集有两套独立实现：

| 系统 | 位置 | Profiler | 数据解析 | Trial 数 | 算子代码 |
|------|------|----------|----------|----------|----------|
| `inner/` | `cann-bench-dev/inner/baseline_perf_prof/scripts/` | ACL `acl.prof` + msprof | Pattern 区间提取 (op_summary CSV) | 20 | `refs/level{1-4}.py` 手写 |
| `run_evaluation` | `cann-bench/src/kernel_eval/eval/perf_eval.py` | `torch_npu.profiler` Level1/Level2 | `KernelDetailsStrategy` (kernel_details CSV) | 5 (warmup=3) | 候选算子 (submission wheel) |

两套系统的**Profiler API、数据解析口径、采集参数**完全不同，导致同一算子同一 case 测出的 `elapsed_us` 可能不一致。baseline 校准值和评测评分的口径脱节会影响评分公平性。

**目标**：设计一个 `scripts/collect_baseline.py` 脚本，**复用评测体系的性能采集逻辑**（`PerfEvaluator` + `KernelDetailsStrategy`），同时**迁移 inner 的 NPU 参考算子代码**（`refs/level{1-4}.py` + `ref_registry.py` + `inputs.py`）到 `scripts/baseline/`，不侵入 `src/kernel_eval/` 代码。

> **v3 更新说明**：v2 设计错误地将 `golden.py` 作为 baseline 参考算子。实际上 **`golden.py` 和 `refs/` 是两套不同的实现**，用途不同：
>
> | 文件 | 用途 | 执行环境 | Kernel topology |
> |------|------|----------|-----------------|
> | `golden.py` | **精度对比**（Golden 参考输出） | CPU / fp64（避免 NPU 溢出） | 多个独立 kernel（Mul + Add + Exp 等） |
> | `refs/level{1-4}.py` | **性能 baseline 采集**（NPU 生产级参考） | NPU / 原始精度 | fused kernel（AutomaticBufferFusionOp、npu_swiglu 等） |
>
> 以 `exp` 算子为例：
> - `golden.py`: `torch.exp(scale * x + shift)` → Mul + Add + Exp（3 个独立 kernel）
> - `refs/level1.py`: `exp_ref` 用 `torch.compile` + torchair → `AutomaticBufferFusionOp`（1 个 fused kernel）
>
> 用 `golden.py` 做 baseline 采集会导致 **kernel topology 与生产环境不一致**，baseline 值虚高（拆开的多个小 kernel vs 一个 fused 大 kernel），评分基准错误。
>
> 因此 v3 改为：将 `cann-bench-dev/inner/baseline_perf_prof/scripts/` 下的 `ref_registry.py`、`refs/level{1-4}.py`、`inputs.py` **迁移到 `scripts/baseline/`**，脚本通过 `sys.path` import 使用迁移后的代码。

---

## 2. 设计原则

| 原则 | 说明 |
|------|------|
| **口径一致** | 采集方式与 `run_evaluation` 完全一致（同一 `PerfEvaluator` + `KernelDetailsStrategy`），baseline 值和评分用的 `elapsed_us` 来自同一把尺子 |
| **不侵入 src** | 脚本通过 import 使用 `src/kernel_eval` 的公开类，但不修改 `src/` 下任何文件 |
| **迁移 refs** | 将 `inner/baseline_perf_prof/scripts/` 下的 `ref_registry.py`、`refs/`、`inputs.py` 迁移到 `scripts/baseline/`，与采集脚本同层，ref_registry.py 保持 auto-discovery 不变 |
| **优先 metadata JSON** | 产出 `metadata/<hardware>.json`（`BaselineStore` 直接可加载），为默认输出模式；可选回填 `cases.yaml` |
| **单命令执行** | 一条命令即可采集指定算子/级别的 baseline，无需手动组合多个脚本 |

---

## 3. 整体架构

```
scripts/collect_baseline.py
┌─────────────────────┐    ┌─────────────────────────────────────────┐
│ ref_registry.py     │    │ src/kernel_eval/                       │
│ + refs/level{1-4}.py│    │   eval/perf_eval.py   (PerfEvaluator) │
│ + inputs.py         │    │   eval/op_runner.py   (OpRunner)      │
│ (NPU 参考算子代码)   │    │   base/perf_strategy.py               │
│ (从 inner 迁移而来) │    │   data/data_generator.py              │
│                     │    │   config.py / device_manager.py        │
│                     │    │   benches/cann_loader.py               │
└─────────────────────┘    │   (性能采集 + 数据加载逻辑)             │
         │                 └─────────────────────────────────────────┘
         ▼                                 │
  ref_fn(inputs, attrs)                    │
         │                                 ▼
         │               OpRunner.run(ref_fn, params,
         │               case_id_str, inputs,
         │               to_device=True, enable_profiler=True)
         │                                 │
         │                                 ▼
         │               PerfEvaluator.run_profiled()
         │                                 │
         │                                 ▼
         │               KernelDetailsStrategy.parse()
         │                                 │
         │                                 ▼
         │               PerfResult.elapsed_us + metadata
         │                                 │
┌──────────────────────────────────────────▼──────────────────────────┐
│ Output Writer                                                        │
│ ├─ metadata/<hardware>.json  (默认，BaselineStore 可加载)           │
│ └─ cases.yaml patch          (可选 --patch-yaml 回填)               │
└─────────────────────────────────────────────────────────────────────┘
```

> **架构说明**：核心路径是 `ref_registry.get_ref()` 加载 NPU 参考实现 → `inputs.py` 构建输入 + NPU 别名处理 → `OpRunner.run()` 执行 ref 函数（NPU 模式 + profiler）→ `PerfResult.elapsed_us` 作为 baseline 值。与 `run_evaluation` 评测 AI 算子的差异是 **func 不同**：评测用 AI 算子，采集用 NPU ref。`OpRunner.run()` 本身是通用接口，接受任何 callable。

---

## 4. 复用策略

### 4.1 refs 迁移：从 inner 复制到 `scripts/baseline/`

```bash
# 迁移命令（一次性操作）
mkdir -p scripts/baseline/refs

# 复制核心文件
cp ../cann-bench-dev/inner/baseline_perf_prof/scripts/ref_registry.py scripts/baseline/
cp ../cann-bench-dev/inner/baseline_perf_prof/scripts/inputs.py        scripts/baseline/
cp ../cann-bench-dev/inner/baseline_perf_prof/scripts/refs/level1.py   scripts/baseline/refs/
cp ../cann-bench-dev/inner/baseline_perf_prof/scripts/refs/level2.py   scripts/baseline/refs/
cp ../cann-bench-dev/inner/baseline_perf_prof/scripts/refs/level3.py   scripts/baseline/refs/
cp ../cann-bench-dev/inner/baseline_perf_prof/scripts/refs/level4.py   scripts/baseline/refs/
```

迁移后 `ref_registry.py` 保持原有 auto-discovery 逻辑（扫描 `refs/*.py`），无需修改——它通过 `Path(__file__).resolve().parent / "refs"` 定位 refs 目录，迁移后路径自然对齐。

**迁移注意事项**：
- `ref_registry.py` 的 `_load()` 函数使用 `Path(__file__).resolve().parent / "refs"` 定位 refs，迁移后自动适配新路径
- `inputs.py` 的 `_apply_op_aliases` 和 `apply_npu_op_aliases` 保留原样——这些算子特有的输入别名处理是 refs 的一部分
- `inputs.py` 的种子 `0xC0FFEE + case_id * 31337` 与评测体系的 `eval_seed` 不同（见 §6.2）
- **后续维护**：新增算子只需在 `scripts/baseline/refs/` 对应 level 文件的 `REGISTRY` 中添加条目，`ref_registry.py` 无需修改

### 4.2 性能采集：直接 import `PerfEvaluator` + `OpRunner`

```python
# scripts/collect_baseline.py
import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kernel_eval.eval.perf_eval import PerfEvaluator
from kernel_eval.eval.op_runner import OpRunner
from kernel_eval.config import Config
from kernel_eval.utils.device_manager import DeviceManager, DeviceConfig
from kernel_eval.base.perf_strategy import KernelDetailsStrategy
from kernel_eval.data.data_generator import DataGenerator
from kernel_eval.utils.param_builder import ParamBuilder
```

**方案选择**：有两种采集路径——

| 方案 | 说明 | 优点 | 缺点 |
|------|------|------|------|
| **A. 使用 `OpRunner.run()`** | 构建输入 → `OpRunner.run(ref_fn, params, case_id, inputs, to_device=True, enable_profiler=True)` | 完全复用执行+profiling路径，OpRunner 内部调用 `PerfEvaluator.run_profiled()` | 需要 DeviceManager、OpRunner 完整初始化；需将 ref_fn 包装为 `(inputs, attrs) → outputs` 签名适配 OpRunner 的调用方式 |
| **B. 使用 `PerfEvaluator.run_profiled()`** | 构建输入 → 包装 ref_fn → 直调 `PerfEvaluator.run_profiled(case_id, func, *args)` | 更轻量 | 需自行处理输入构建、to_device、profiler session |

**推荐方案 A**。理由：
- `OpRunner.run()` 已封装了 to_device、profiler 启用、输出收集等逻辑
- 与 `run_evaluation` 的执行路径完全一致（同一条代码管线）
- ref_fn 的签名是 `(inputs, attrs) -> outputs`，而 OpRunner 期望 `func(*args, **kwargs)`，需要一个 wrapper（见 §4.3）

### 4.3 参考算子代码：import 迁移后的 `ref_registry.py`

```python
# scripts/collect_baseline.py
REFS_DIR = PROJECT_ROOT / "scripts" / "baseline"
sys.path.insert(0, str(REFS_DIR))

import ref_registry
ref_fn = ref_registry.get_ref("level2/cummin")
```

`ref_registry.py` 自动加载 `refs/level{1-4}.py` 中的 `REGISTRY`，返回 `ref_fn(inputs, attrs) -> outputs` callable。

**ref_fn 签名适配**：`OpRunner.run()` 期望 `func(*args, **kwargs)` 形式的 callable，但 `ref_fn` 的签名是 `(inputs, attrs) -> outputs`（inputs 是结构化的 list/tensor-list）。需要一个 wrapper：

```python
# 签名适配 wrapper（与 bench_baseline.py 的 GenericRefModule 类似）
def ref_fn_wrapper(ref_fn, attrs, *flat_args):
    """将 (inputs, attrs) 签名的 ref_fn 包装为 (*flat_args) 签名"""
    # flat_args 可能需要根据结构 unflatten
    # 简单情况下直接传 inputs=list(flat_args) + attrs
    inputs = list(flat_args)
    return ref_fn(inputs, attrs)
```

更完整的方案是复用 `bench_baseline.py` 的 `_flatten_with_structure` + `GenericRefModule` 逻辑——将嵌套输入 flatten/unflatten。

### 4.4 输入构建：两套可选路径

| 优先路径 | 说明 |
|----------|------|
| **A. 复用 `DataGenerator`** | import `kernel_eval.data.DataGenerator`，直接用评测体系的数据生成器，与 `run_evaluation` 的输入生成方式一致 |
| **B. 复用 `inputs.py`** | import 迁移后的 `inputs.py`（`build_inputs` / `to_device` / `apply_npu_op_aliases`），保留 inner 的输入构建逻辑 |

**推荐路径 B**（用 `inputs.py`）。理由：
- `inputs.py` 的 `build_inputs` 是 refs 体系的配套输入构建器，与 ref_fn 的输入格式完全对齐
- `inputs.py` 的 `apply_npu_op_aliases` 做了算子特有的 NPU 输入兼容调整（如 `quant_matmul` 的 `npu_trans_quant_param` 预打包、`mla` 的 v=k_nope 别名、`sparse_flash_attention` 的 value=key[:Dv] 别名），这些逻辑是 refs 的一部分，不是评测体系的一部分
- `inputs.py` 的种子 `0xC0FFEE + case_id * 31337` 与 inner baseline 采集一致，便于跨系统比对

**实际方案**：用 `inputs.py` 构建输入 + NPU 别名处理，但**可选**用 `DataGenerator` 作为替代（当 `inputs.py` 与某个 case 的 dtype/value_range 解析有差异时）。

```python
# scripts/collect_baseline.py
import inputs as bench_inputs  # 迁移后的 inputs.py

inputs_cpu = bench_inputs.build_inputs(
    case["input_shape"], case["dtype"],
    case.get("value_range"), case_id, op_key=op_path,
)
inputs_npu = bench_inputs.to_device(inputs_cpu, f"npu:{device_id}")
inputs_npu = bench_inputs.apply_npu_op_aliases(inputs_npu, op_path, attrs)
```

### 4.5 Case 发现：复用 `CannCaseLoader` + `CannTaskLoader`

```python
from kernel_eval.registry.loader_registry import get_case_loader, get_task_loader

case_loader = get_case_loader("cann", tasks_root=str(TASKS_ROOT))
task_loader = get_task_loader("cann", tasks_root=str(TASKS_ROOT))

# 获取指定算子的所有 cases
cases = case_loader.scan_by_operator("Cummin")  # → List[CannCaseSpec]

# 获取算子定义信息
op_info = task_loader.get_operator("level2/cummin")  # → CannTaskSpec
```

`CannCaseSpec` 包含 `input_shapes`、`dtypes`、`value_ranges`、`attrs`、`case_id`、`rel_path` 等字段，是 cases.yaml 的结构化表示。

但 **`inputs.py` 的 `build_inputs` 直接接受 cases.yaml 的原始 dict 字段**（`input_shape`、`dtype`、`value_range`），而不是 `CannCaseSpec`。因此实际实现中有两种路径：
1. 用 `CannCaseLoader` 发现算子列表 → 从 `cases.yaml` 直接读取 case dict → `inputs.build_inputs(case_dict, ...)`
2. 用 `CannCaseLoader` 获取 `CannCaseSpec` → 从 `CannCaseSpec` 提取字段 → `inputs.build_inputs(...)`

推荐路径 1（更简单，与 `bench_baseline.py` 的输入读取方式一致）。

### 4.6 不侵入 src 的保障

| 保障机制 | 说明 |
|----------|------|
| **只 import，不修改** | 脚本只 import `src/kernel_eval` 的公开类，不 patch 任何行为 |
| **OpRunner.run() 天然通用** | `run(func, params, case_id, inputs, ...)` 接受任何 callable，不需要为 baseline 采集新增入口 |
| **独立 sys.path** | 脚本在运行时通过 `sys.path.insert` 引入 src 和 refs，不污染 src 的 import 结构 |
| **输出格式独立** | 写 `metadata/<hardware>.json`，与 `src/kernel_eval` 的数据流完全独立 |

---

## 5. 脚本设计

### 5.1 命令行接口

```bash
# 采集所有 level1 算子的 baseline
python scripts/collect_baseline.py --level 1

# 采集单个算子（按 rel_path）
python scripts/collect_baseline.py --op level2/cummin

# 采集单个算子（按算子名，自动匹配）
python scripts/collect_baseline.py --op Cummin

# 指定 case_id
python scripts/collect_baseline.py --op level2/cummin --cases 1,5,13

# 采集所有级别
python scripts/collect_baseline.py --all

# 指定硬件和设备
python scripts/collect_baseline.py --level 1 --device-id 7

# 采集参数（warmup/repeat 对齐 run_evaluation 默认值）
python scripts/collect_baseline.py --op level1/exp --warmup 3 --repeat 5

# 高精度采集（更多 trial，用于校准）
python scripts/collect_baseline.py --op level1/exp --warmup 5 --repeat 20

# 仅输出到 metadata JSON（默认行为）
python scripts/collect_baseline.py --level 1

# 同时回填到 cases.yaml
python scripts/collect_baseline.py --level 1 --patch-yaml

# dry-run（只打印计划，不执行）
python scripts/collect_baseline.py --level 1 --dry-run

# 指定评测集根目录
python scripts/collect_baseline.py --op level1/exp --bench-root /path/to/tasks
```

### 5.2 核心流程伪代码

```python
class BaselineCollector:
    """Baseline 性能采集器"""

    def __init__(self, config: Config, bench_root: Path):
        self.config = config
        self.bench_root = bench_root

        # 初始化评测体系组件（复用 Evaluator 的初始化逻辑）
        self.device_manager = DeviceManager(DeviceConfig(
            type=config.device_type, device_id=config.device_id,
        ))
        self.perf_evaluator = PerfEvaluator(
            config=config, device_manager=self.device_manager,
            warmup=config.warmup, repeat=config.repeat,
            archive_prof=True,
        )
        self.op_runner = OpRunner(self.device_manager, self.perf_evaluator)

        # 数据加载器（Case 发现 + 算子信息）
        self.case_loader = get_case_loader("cann", tasks_root=str(bench_root))
        self.task_loader = get_task_loader("cann", tasks_root=str(bench_root))

    def collect_one_case(self, op_path: str, case: dict) -> dict:
        """采集单个 case 的 baseline 性能数据

        Args:
            op_path: 算子路径，如 "level2/cummin"
            case: cases.yaml 中的单个 case dict
                   （含 input_shape, dtype, value_range, attrs, case_id）
        """

        case_id = int(case["case_id"])
        case_id_str = f"{op_path}_{case_id}"
        attrs = _normalize_attrs(case.get("attrs") or {})

        # 1. 获取 ref 函数
        ref_fn = ref_registry.get_ref(op_path)
        if ref_fn is None:
            return {"op_path": op_path, "case_id": case_id,
                    "skipped": True, "error": f"no ref registered for {op_path}"}

        # 2. 构建输入（使用 inputs.py — refs 体系的配套输入构建器）
        try:
            inputs_cpu = bench_inputs.build_inputs(
                case["input_shape"], case["dtype"],
                case.get("value_range"), case_id, op_key=op_path,
            )
            inputs_npu = bench_inputs.to_device(
                inputs_cpu, f"npu:{self.config.device_id}"
            )
            inputs_npu = bench_inputs.apply_npu_op_aliases(
                inputs_npu, op_path, attrs
            )
            flat, structure = _flatten_with_structure(inputs_npu)
        except Exception as e:
            return {"op_path": op_path, "case_id": case_id,
                    "elapsed_us": None, "error_msg": f"input_build_FAIL: {e}"}

        # 3. 包装 ref 为 OpRunner 可接受的 callable
        model = GenericRefModule(ref_fn, attrs, structure).npu()

        # 4. 执行 ref 函数 + profiler
        run_result = self.op_runner.run(
            model, flat, case_id_str,
            inputs_npu, to_device=True, enable_profiler=True,
        )

        # 5. 汇总结果
        if not run_result.success:
            return {
                "op_path": op_path, "case_id": case_id,
                "elapsed_us": None, "error_msg": run_result.error,
            }

        perf_result = run_result.perf_result
        return {
            "op_path": op_path, "case_id": case_id,
            "elapsed_us": perf_result.elapsed_us,
            "aicore_e2e": perf_result.metadata.get("aicore_e2e"),
            "aicpukernel_gap": perf_result.metadata.get("aicpukernel_gap"),
            "aicore_e2e_jitter": perf_result.metadata.get("aicore_e2e_jitter"),
            "device_kernels": perf_result.op_times.get("device_kernels", {}),
            "data_source": perf_result.metadata.get("data_source"),
            "error_msg": perf_result.error_msg,
        }
```

> **与 bench_baseline.py 的关键差异**：`bench_baseline.py` 使用 `AdvancedPerformanceEngine`（ACL profiling + msprof），而 `collect_baseline.py` 使用 `OpRunner.run()` → `PerfEvaluator.run_profiled()`（`torch_npu.profiler` + `KernelDetailsStrategy`）。这是**口径统一的根本改进**——两套 profiler 的数据解析方式不同，现在统一为评测体系的标准口径。

### 5.3 Case 发现与批量采集

```python
def discover_ops(self, op_filter=None, level_filter=None):
    """发现需要采集的算子列表"""

    # 从 CannTaskLoader 获取所有算子
    all_ops = self.task_loader.list_operators()  # → List[CannTaskSpec]

    # 按 level 过滤
    if level_filter:
        all_ops = [op for op in all_ops
                   if op.rel_path.startswith(f"level{level_filter}/")]

    # 按 op 过滤
    if op_filter:
        all_ops = [op for op in all_ops
                   if op.rel_path == op_filter or op.name == op_filter]

    return all_ops


def collect_op(self, op_path: str, cases_filter: set = None):
    """采集单个算子的所有 case"""

    # 加载 cases.yaml（直接读取，兼容 inputs.py 的字段格式）
    cases_yaml = self.bench_root / op_path / "cases.yaml"
    with open(cases_yaml) as f:
        cases = yaml.safe_load(f)["cases"]

    if cases_filter:
        cases = [c for c in cases if int(c["case_id"]) in cases_filter]

    # 检查是否有 ref
    ref_fn = ref_registry.get_ref(op_path)
    if ref_fn is None:
        print(f"[SKIP] {op_path}: no ref registered")
        return []

    results = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] Collecting {op_path}_{case['case_id']}...")
        result = self.collect_one_case(op_path, case)
        results.append(result)

        if result.get("elapsed_us") is not None:
            print(f"  ✅ elapsed_us={result['elapsed_us']:.2f}μs")
        else:
            print(f"  ❌ {result.get('error_msg', 'unknown error')}")

        # 清理内存
        try:
            import torch_npu
            torch_npu.npu.empty_cache()
        except Exception:
            pass
        import gc
        gc.collect()

    return results
```

### 5.4 输出格式

#### metadata JSON（默认输出，与 BaselineStore 兼容）

```json
{
  "_metadata": {
    "description": "CANN baseline 性能数据（collect_baseline.py 采集）",
    "hardware": "910b2",
    "generated_at": "2026-06-06T15:30:00",
    "source": "collect_baseline.py (PerfEvaluator + KernelDetailsStrategy + refs)",
    "warmup": 5,
    "repeat": 20,
    "profiler_level": "Level1",
    "input_builder": "inputs.py (seed=0xC0FFEE+case_id*31337)",
    "collection_method": "ref_func_profiling"
  },
  "level2": {
    "cummin": {
      "1": { "baseline_perf_us": 33429.0, "t_hw_us": 1.09 },
      "2": { "baseline_perf_us": 14.58, "t_hw_us": 8.74 }
    }
  }
}
```

与现有 `tasks/metadata/910b2.json` 格式完全一致，`BaselineStore` 可以直接加载。

> **注意**：`t_hw_us` 来自现有 metadata JSON（或旧版 cases.yaml），不是采集产出。脚本采集时从 `BaselineStore` / `CannCaseSpec` 读取 `t_hw_us`，写入 metadata JSON 时一并携带。新增采集的 case 只更新 `baseline_perf_us`，不修改已有的 `t_hw_us`。

#### cases.yaml 回填（可选 `--patch-yaml`）

通过 regex-based 行级编辑（复用 `apply_baselines.py` 的 patch 逻辑），将 `baseline_perf_us` 回填到 `cases.yaml`：

- 使用 `ruamel.yaml` 保留注释和格式的行级编辑
- 保留 `baseline_source: auto` + `baseline_updated_at` 时间戳标记
- 保留 `baseline_source: manual` 的 case 不覆盖（需 `--force`）
- 回填后重新生成 `cases.csv`（调用 `scripts/utils/yaml_to_csv.py`）

#### 采集报告（人类可读）

输出表格，类似 inner 的 `level{N}_baseline_compare.txt`：

```
op                               cid  baseline_us  t_hw_us  elapsed_us  elapsed/base  k/call  kernels
-----------------------------------------------------------------------------------------------
cummin                            1     33429.0      1.09     33430.5      1.00x       2  cummin_custom×1+Cast×1
cummin                            2       14.58      8.74      14.60      1.00x       2  Cummin×1+Cast×1
```

---

## 6. 关键细节

### 6.1 Trial 数量选择

`run_evaluation` 默认 warmup=3, repeat=5。baseline 采集需要更稳定的值，建议：

| 场景 | warmup | repeat | 说明 |
|------|--------|--------|------|
| **快速校验** | 3 | 5 | 与 run_evaluation 一致，用于快速比对 |
| **标准校准** | 5 | 20 | 接近 inner 的 20 trial，统计稳定性更好 |

脚本默认使用 **标准校准** (warmup=5, repeat=20)，用户可通过 `--warmup` / `--repeat` 调整。

### 6.2 `inputs.py` 与 `DataGenerator` 的种子差异

两者使用不同的确定性种子机制：

| 来源 | 种子公式 | 用途 |
|------|----------|------|
| `inputs.py` | `torch.Generator().manual_seed(0xC0FFEE + case_id * 31337)` | inner baseline 采集，保持与 inner 比对一致 |
| `DataGenerator` | `eval_seed=0` → `SHA256(case_id_str) % 2^31` | `run_evaluation` 评测，保持与评测一致 |

**建议**：baseline 采集使用 `inputs.py` 的种子机制（保持与 inner 莹 baseline 比对一致），但 `_metadata` 中记录种子信息便于追溯。

> **开放问题**：是否应该统一为 `DataGenerator` 的种子？统一后输入数据与评测完全一致，但会导致与 inner baseline 比对时输入不同。见 §14 Q6。

### 6.3 `apply_npu_op_aliases` 处理

`inputs.py` 的 `apply_npu_op_aliases` 做了算子特有的 NPU 输入兼容调整。涉及的算子：

| 算子 | 别名处理 | 说明 |
|------|----------|------|
| `level3/quant_matmul` | `_quant_matmul_prepack_scale` | fp32 scale → int64 prepack（`npu_trans_quant_param`），避免 TransQuantParamV2 kernel 混入 baseline 时间 |

此外，`inputs.py` 的 `_apply_op_aliases`（CPU 侧别名）处理了更多算子：

| 算子 | CPU 侧别名 | 说明 |
|------|------------|------|
| `level4/mla` | `v == k_nope` | KV cache 共享 |
| `level4/sparse_flash_attention` | `value == key[..., :Dv]` | KV cache 共享 |
| `level3/roi_align` | boxes[:, 0] = arange(N) % B | batch_idx 约束 |
| `level3/moe_re_routing` | expert_token_num_per_rank 分区 | Sum == A 约束 |

这些逻辑已包含在迁移后的 `inputs.py` 中，`collect_baseline.py` 只需调用 `build_inputs(op_key=op_path)` → `apply_npu_op_aliases(op_path, attrs)` 即可。

### 6.4 `t_hw_us` 来源

`t_hw_us`（硬件理论下限）来自手动标定，不是采集产出。脚本采集时：

- **metadata JSON 已有**：从 `BaselineStore.get_t_hw()` 读取，写入新 metadata 时保留
- **metadata JSON 无**（新增 case）：`t_hw_us` 设为 0.0（需后续手动标定）
- **cases.yaml 有**（回填模式）：从 cases.yaml 读取 `t_hw_us`

### 6.5 Profiler 数据归档

与 `run_evaluation` 行为一致：profiling 中间数据归档到 `reports/prof_data/baseline/<op_path>/<case_id>/`，便于事后诊断。

归档路径通过 `Config.reports_dir` + `PerfEvaluator.prof_data_dir` 控制。

### 6.6 确定性种子

两种选择（见 §14 Q6）：
- **A. `inputs.py` 种子**：`0xC0FFEE + case_id * 31337` — 与 inner baseline 比对一致
- **B. `DataGenerator` 种子**：`SHA256(case_id_str) % 2^31` — 与评测一致

### 6.7 防作弊机制

baseline 采集不需要 `Evaluator.evaluate_case` 的防作弊机制：

- **TorchOpGuard**：ref 函数是可信参考实现，不需要监听
- **accuracy_retry**：不需要二次验证精度
- **InputPool**：不需要防 data_ptr 缓存攻击

脚本设置 `torch_op_guard_mode = "off"`、`enable_accuracy_retry = False`。

### 6.8 Ref 覆盖范围

inner refs 的 REGISTRY 覆盖情况：

| Level | 覆盖算子数 | 未覆盖算子 | 说明 |
|-------|-----------|-----------|------|
| level1 | 8/8 | 0 | 全覆盖（exp, gelu, sigmoid, mish, masked_scale, swi_glu, foreach_norm, foreach_addcdiv_scalar） |
| level2 | 16/16 | 0 | 全覆盖 |
| level3 | 17/~20 | dilation_2d, engram_gate_fusion, mhc_sinkhorn, strided_slice | 无 torch/torch_npu 等价实现或语义分歧太大 |
| level4 | 8/8 | 0 | 全覆盖（gqa, grouped_matmul_swiglu_quant, gru, lstm, mha, mla, mla_prolog, sparse_flash_attention） |

未覆盖的算子 `ref_registry.get_ref()` 返回 `None`，脚本 SKIP 并记录 warning。

---

## 7. 目录布局

```
scripts/
├── collect_baseline.py          # 主脚本（本次新增）
├── baseline/                    # refs 迁移目录（从 inner 复制而来）
│   ├── ref_registry.py          # 迁移：参考算子注册表（auto-discovery）
│   ├── inputs.py                # 迁移：输入构建 + NPU 别名处理
│   └── refs/
│       ├── __init__.py          # 迁移：空 __init__
│       ├── level1.py            # 迁移：level1 NPU 参考实现（8 个算子）
│       ├── level2.py            # 迁移：level2 NPU 参考实现（16 个算子）
│       ├── level3.py            # 迁移：level3 NPU 参考实现（17 个算子）
│       └── level4.py            # 迁移：level4 NPU 参考实现（8 个算子）
├── migrate_baseline_to_data.py  # 已有：cases.yaml → metadata JSON 迁移
├── run_evaluation.sh            # 已有：评测入口
├── utils/
│   ├── yaml_to_csv.py           # 已有：cases.yaml → cases.csv 转换
│   └── ...
└── ...

tasks/                           # 不修改任何文件（除可选 --patch-yaml）
├── metadata/
│   ├── 910b2.json               # 输出：baseline JSON（BaselineStore 加载）
│   └── ...
├── level1/
│   ├── exp/
│   │   ├── cases.yaml           # 数据源：case 定义
│   │   ├── golden.py            # 参考：精度对比（CPU 侧，不用于 baseline 采集）
│   │   ├── proto.yaml           # 数据源：算子定义
│   │   └── ...
│   └── ...
└── ...

src/kernel_eval/                 # 不修改任何文件
├── eval/
│   ├── perf_eval.py             # import: PerfEvaluator
│   ├── op_runner.py             # import: OpRunner
│   ├── evaluator.py             # 参考: Evaluator 初始化+执行流程
│   └── ...
├── base/
│   ├── perf_strategy.py         # import: KernelDetailsStrategy
│   ├── result.py                # import: PerfResult
│   └── ...
├── data/
│   ├── data_generator.py        # 可选 import: DataGenerator（替代输入构建）
│   └── ...
├── utils/
│   ├── device_manager.py        # import: DeviceManager
│   ├── baseline_store.py        # import: BaselineStore
│   ├── baseline_resolver.py     # import: DEFAULT_HARDWARE
│   └── ...
├── registry/
│   ├── loader_registry.py       # import: get_case_loader, get_task_loader
│   └── ...
├── benches/
│   ├── cann_loader.py           # import: CannCaseLoader, CannTaskLoader
│   └── ...
└── config.py                    # import: Config

../cann-bench-dev/inner/baseline_perf_prof/scripts/  # 源：refs 迁移来源
├── ref_registry.py              # → scripts/baseline/ref_registry.py
├── inputs.py                    # → scripts/baseline/inputs.py
├── bench_baseline.py            # 不迁移（旧脚本，后续废弃）
├── perf_engine.py               # 不迁移（旧 profiler engine，用 PerfEvaluator 替代）
├── refs/
│   ├── level1.py                # → scripts/baseline/refs/level1.py
│   ├── level2.py                # → scripts/baseline/refs/level2.py
│   ├── level3.py                # → scripts/baseline/refs/level3.py
│   └── level4.py                # → scripts/baseline/refs/level4.py
└── ...
```

---

## 8. 与现有系统的关系

### 8.1 inner 的 `bench_baseline.py`

`collect_baseline.py` **替代** `bench_baseline.py` 的功能，但不删除它。过渡期两套并存，后续 inner 可逐步废弃 `bench_baseline.py`。

| 替代关系 | inner 旧脚本 | 新脚本 |
|----------|--------------|--------|
| 算子代码 | refs/level{1-4}.py + ref_registry（原地引用） | refs/level{1-4}.py + ref_registry（迁移到 scripts/baseline/） |
| Profiler | ACL acl.prof + msprof (perf_engine.py) | torch_npu.profiler Level1/Level2 (PerfEvaluator) |
| 数据解析 | Pattern 区间 (op_summary) | KernelDetailsStrategy (kernel_details CSV) |
| 输入构建 | inputs.py（原地引用） | inputs.py（迁移到 scripts/baseline/） |
| Case 发现 | 手动枚举 (LEVEL1_OPS 等硬编码列表) | CannTaskLoader + CannCaseLoader（自动发现） |
| 输出格式 | baseline_perf_<soc>_<ts>.json | metadata/<hardware>.json + 可选 cases.yaml 回填 |

### 8.2 `apply_baselines.py` (inner/tests/)

`apply_baselines.py` 的 cases.yaml 回填逻辑可被 `collect_baseline.py` 的 `--patch-yaml` 模式替代。

### 8.3 `migrate_baseline_to_data.py` (scripts/)

`migrate_baseline_to_data.py` 的 extract 模式从 cases.yaml 抽取已有 baseline 值到 metadata JSON。`collect_baseline.py` 的默认模式直接产出 metadata JSON，**不需要先写 cases.yaml 再抽取**——流程更短。

两者的关系：
- `migrate_baseline_to_data.py`：**迁移**（从 cases.yaml 搬到 metadata JSON，数据不变）
- `collect_baseline.py`：**采集**（重新执行 ref 函数 + profiler，产生新 baseline 值）

### 8.4 `update_baseline_perf.py` (inner/)

`update_baseline_perf.py` 从 test_results.json 回填 cases.yaml。`collect_baseline.py` 直接采集 + 输出，不再需要中间 JSON 转储。

### 8.5 `perf_engine.py` (inner/)

`perf_engine.py`（`AdvancedPerformanceEngine`）是 inner 的旧 profiler engine（ACL profiling + msprof），不迁移。`collect_baseline.py` 使用 `PerfEvaluator` + `KernelDetailsStrategy` 替代，口径统一。

---

## 9. 错误处理与容错

### 9.1 算子执行失败

| 场景 | 处理 | 输出 |
|------|------|------|
| ref 未注册 | SKIP，记录 warning | `{skipped: True, error: "no ref registered"}` |
| ref 返回 None（如 dtype 不支持） | SKIP，记录 warning | `{elapsed_us: None, error_msg: "ref returned None"}` |
| 输入构建失败 | SKIP，记录 error | `{elapsed_us: None, error_msg: "input_build_FAIL"}` |
| ref 执行异常（shape 不对 / NPU 异常） | SKIP，记录 error + traceback | `{elapsed_us: None, error_msg: ...}` |
| profiler 采集失败（CSV 缺失 / 解析异常） | SKIP，记录 error | `{elapsed_us: None, error_msg: ...}` |
| NPU 设备异常 | 渐进恢复（与 Evaluator.evaluate_operator 一致），恢复失败则跳过剩余 case | `{error_msg: "device unrecoverable"}` |

### 9.2 部分采集中断

脚本支持**断点续采**：
- 每采集完一个 case，立即将结果追加到 metadata JSON
- 中断后重新运行时，已有结果的 case 自动跳过（`--skip-existing`）
- 可通过 `--force-recollect` 强制重新采集所有 case

### 9.3 无 NPU 环境

无 NPU 环境（mac / 无 torch_npu）时脚本**优雅退出**：
- 检测 `torch_npu` 是否可导入
- 不可导入时打印明确提示并 exit(1)，不 crash
- CPU 模式不支持 profiler，无法采集 baseline 性能数据

### 9.4 内存管理

与 `Evaluator` 一致：
- 每个 case 完成后执行 `torch_npu.npu.empty_cache()` + `gc.collect()`
- 释放 outputs tensor
- 大 shape case 注意 OOM 风险

---

## 10. 并发与资源管理

### 10.1 单进程串行采集

默认单进程串行采集，与 `run_evaluation` 的 profiler 单线程约束一致（`torch_npu.profiler` 内部使用全局单例 `ProcessPoolExecutor`，多线程并发会导致 Bus error）。

### 10.2 多设备并行

可通过 `--device-ids 0,1,2,3` 指定多个 NPU 设备，每个设备独立进程：
- 主进程将算子列表分发到多个子进程（每个绑定不同 `device_id`）
- 子进程各自串行采集，主进程汇总结果

> **注意**：多进程并行采集暂不实现（P3），初期只支持单设备串行。

### 10.3 超时控制

与 `Evaluator.evaluate_from_source` 一致：
- 单 case 超时：`--case-timeout`（默认 240s）
- 单算子超时：`--op-timeout`（默认 600s）

---

## 11. 实现优先级

| 优先级 | 功能 | 说明 |
|--------|------|------|
| P0 | refs 迁移 | 将 inner refs/inputs/ref_registry 复制到 scripts/baseline/ |
| P0 | 单算子采集 | `--op level2/cummin` 或 `--op Cummin`，最核心功能 |
| P0 | metadata JSON 输出 | 默认输出，与 BaselineStore 对齐 |
| P0 | Case 发现 | CannCaseLoader + CannTaskLoader，自动发现 cases |
| P1 | 全级别采集 | `--level 1` / `--all`，批量采集 |
| P1 | cases.yaml 回填 | `--patch-yaml`，自动回填 |
| P1 | 断点续采 | `--skip-existing`，跳过已采集的 case |
| P2 | 采集报告 | 人类可读的表格输出 |
| P2 | dry-run | `--dry-run`，列出计划但不执行 |
| P2 | 多硬件 dict baseline | `baseline_perf_us: {910b2: 40.2, 910b1: 45.1}` |
| P3 | 多设备并行 | `--device-ids 0,1,2,3`，多进程并行采集 |
| P3 | 超时控制 | `--case-timeout` / `--op-timeout` |

---

## 12. 验收标准

1. **口径一致性**：同一算子同一 case，`collect_baseline.py` 的 `elapsed_us` 与 `run_evaluation` 使用同一 `KernelDetailsStrategy` 解析同一 profiler 输出格式，数值偏差 < 5%（允许 warmup/repeat 参数不同带来的正常波动）
2. **BaselineStore 可加载**：输出的 `metadata/<hardware>.json` 可被 `BaselineStore` 直接加载，`get_perf()` / `get_t_hw()` 返回正确值
3. **cases.yaml 回填**：`--patch-yaml` 后 `cases.yaml` 的 `baseline_perf_us` 字段正确更新，`cases.csv` 同步再生
4. **不侵入 src**：`src/kernel_eval/` 下无任何文件被修改
5. **ref_registry 可加载**：`ref_registry.get_ref()` 返回的 callable 与 inner 旧脚本的 ref_fn 一致
6. **refs 覆盖范围**：迁移后的 refs/level{1-4}.py 覆盖与 inner 一致的算子集合
7. **input 构建一致**：`inputs.build_inputs()` 的输入与 inner `bench_baseline.py` 的输入一致（相同种子 → 相同 tensor）
8. **mac 兼容**：无 NPU 环境时脚本优雅退出，不 crash

---

## 13. 测试计划

| 测试类型 | 测试内容 | 方法 |
|----------|----------|------|
| **UT: ref_registry** | `ref_registry.get_ref("level1/exp")` 返回 `exp_ref` callable | import + 检查 REGISTRY |
| **UT: inputs.py** | `build_inputs(shape, dtype, range, case_id)` 输出 tensor 与 inner 一致 | 同种子比对 tensor 值 |
| **UT: 输出格式** | metadata JSON 格式与 BaselineStore 兼容 | `BaselineStore.load()` + `get_perf()` 验证 |
| **UT: t_hw_us 保留** | 采集后 metadata JSON 中已有 t_hw_us 值不变 | 对比采集前后 JSON |
| **UT: GenericRefModule** | wrapper 正确将 `(inputs, attrs)` 签名适配为 `(*flat_args)` | 对比 ref_fn(inputs, attrs) vs model(flat_args) 输出 |
| **E2E: 单算子采集** | `collect_baseline.py --op level1/exp` 完整执行 | 在 NPU 环境执行，验证输出 |
| **E2E: 全级别 dry-run** | `collect_baseline.py --all --dry-run` 列出所有算子+case | 检查输出计划完整性 |
| **E2E: 口径对比** | 同一 case，`collect_baseline.py` vs `run_evaluation` 的 elapsed_us | 偏差 < 5% |
| **E2E: 与 inner 对比** | 同一 case，`collect_baseline.py` vs `bench_baseline.py` 的 ref kernel topology | kernel 名称和数量一致 |
| **E2E: 无 NPU 环境** | `collect_baseline.py` 在无 torch_npu 环境下优雅退出 | 检查 exit code + 输出消息 |
| **E2E: 断点续采** | 中断后重新运行，已有结果不重复采集 | 验证 `--skip-existing` |

---

## 14. 开放问题

| # | 问题 | 选项 | 建议 |
|---|------|------|------|
| Q1 | refs 是否保留 inner 原始目录的副本 | A. 仅迁移到 scripts/baseline/（inner 保留原位，后续废弃）<br>B. 迁移后删除 inner 中的副本 | **A** — 过渡期两套并存 |
| Q2 | `inputs.py` 是否做适配修改 | A. 原样迁移（不改 seed、不改逻辑）<br>B. 统一种子为 DataGenerator 的 SHA256 hash | 初期 **A**，后续考虑 **B** |
| Q3 | `TorchOpGuard` 是否启用 | A. off（ref 是可信实现）<br>B. warn（调试用） | **A** |
| Q4 | cases.yaml 回填是否默认开启 | A. 默认关闭（`--patch-yaml` 可选）<br>B. 默认开启 | **A** — metadata JSON 是主输出 |
| Q5 | 多硬件 baseline 格式 | A. 单硬件 scalar（910b2.json）<br>B. 多硬件 dict | 初期 **A**，后续扩展 B（P3） |
| Q6 | 输入构建种子机制 | A. `inputs.py` 种子（`0xC0FFEE + case_id * 31337`，与 inner 比对一致）<br>B. `DataGenerator` 种子（SHA256 hash，与评测一致） | 初期 **A**，但 `_metadata` 中记录种子信息 |