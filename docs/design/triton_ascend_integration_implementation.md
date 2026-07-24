# CANN Bench Triton-Ascend 集成实现与运行原理

## 1. 结论

CANN Bench 不直接调用 Triton 编译器，也没有新增 `TritonRunner`。它把用户提交的
Triton 算子当成普通 Python callable：安装用户的 `cann_bench` wheel，找到
`cann_bench.exp`，把输入放到 NPU，然后执行：

```python
outputs = func(**updated_params)
```

用户函数内部的 `_exp_kernel[grid](...)` 才是进入 Triton-Ascend JIT 和 NPU launch
的边界。`@triton.jit` 在模块导入时只创建 `JITFunction`，不会立即生成 NPU 二进制。
第一次 launch 某个 specialization 时才查询缓存并按需编译。

这套方案沿用现有 submission 合同：用户提交 `cann_bench` wheel，CANN Bench 调用
公开 Python 函数；Triton-Ascend 的编译和 launch 差异留在用户 wrapper 与平台运行时。

## 2. 端到端调用链

```text
用户源码目录
  |
  | build.sh：只打包 Python 源码，不编译 NPU kernel
  v
cann_bench-1.0.0-py3-none-any.whl
  |
  | pip install --force-reinstall --no-deps
  v
import cann_bench -> 扫描公开 callable -> 匹配 proto.yaml
  |
  | OpRunner: outputs = func(**updated_params)
  v
cann_bench.exp(x, base, scale, shift)
  |
  | _exp_kernel[grid](...)
  v
JITFunction.run
  |-- 进程内 cache 命中 ---------------------------> launch
  `-- 未命中 -> triton.compiler.compile
                  |-- 磁盘 cache 命中 --------------> load + launch
                  `-- 未命中 -> TTIR -> ttadapter -> npubin
                                                -> load -> NPU stream
```

这里存在两个名称相似但性质不同的“编译”：

| 阶段 | 触发位置 | 实际工作 |
|---|---|---|
| submission 构建 | `bash build.sh` | setuptools 把 `.py` 文件封装成 wheel |
| kernel JIT | `_exp_kernel[grid](...)` | Triton-Ascend 为当前 specialization 生成 NPU binary |

## 3. 用户提交什么

最小提交结构如下：

```text
submission/
├── build.sh
├── setup.py
└── cann_bench/
    ├── __init__.py
    └── exp.py
```

提交合同：

1. Python 包名必须是 `cann_bench`。
2. `cann_bench/__init__.py` 必须导出任务函数。
3. 函数名、参数名、默认值和输出语义必须与任务的 `proto.yaml` 一致。
4. 函数接收 NPU Tensor，返回 NPU Tensor；核心数值计算必须在 DSL kernel 内完成。
5. wheel 不携带或安装 `torch`、`torch_npu`、`triton-ascend`；这些由评测镜像提供。
6. 一个 wheel 可以包含多个算子，CANN Bench 会扫描所有公开 callable。

本示例的导出代码是：

```python
from .exp import exp
from .masked_scale import masked_scale
from .mish import mish
from .sigmoid import sigmoid
from .swi_glu import swi_glu

__all__ = ["exp", "masked_scale", "mish", "sigmoid", "swi_glu"]
```

注意：`__all__` 是提交包显式声明的接口合同；当前扫描器实际使用
`dir(cann_bench)`，因此任何非下划线开头的 callable 都可能被发现。

## 4. CANN Bench 如何加载 submission

### 4.1 安装 wheel

本次修改位于 `src/kernel_eval/data/package_manager.py`。所有提交通常都使用
`cann_bench==1.0.0`，所以必须强制覆盖同版本旧包，同时禁止 submission 改写平台依赖：

```python
result = subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "--no-deps",
        whl_path,
    ],
    capture_output=True,
    text=True,
    timeout=60,
)
```

### 4.2 清理旧模块并发现接口

只删除顶层 `cann_bench` 不够，因为 `cann_bench.exp` 等子模块仍可能指向上一份提交。
当前实现会清理整棵模块缓存，再重新 import：

```python
for module_name in list(sys.modules):
    if module_name == "cann_bench" or module_name.startswith("cann_bench."):
        del sys.modules[module_name]
importlib.invalidate_caches()

import cann_bench

for name in dir(cann_bench):
    if name.startswith("_"):
        continue
    attr = getattr(cann_bench, name)
    if callable(attr) and not isinstance(attr, type):
        interfaces.append(
            InterfaceInfo(name=name, callable=attr, signature=...)
        )
```

### 4.3 名称匹配

`OperatorMatcher.load_ai_operator()` 根据任务名构造候选名。例如 `Exp` 会匹配
`exp` 以及 schema 中声明的函数名。它先检查 `torch.ops.cann_bench`，再检查用户模块：

```python
import cann_bench

for name in candidates:
    if hasattr(cann_bench, name):
        func = getattr(cann_bench, name)
        self._ai_op_cache[cache_key] = func
        return func
```

本次 Triton wheel 是纯 Python 包，没有 `.so` 和自定义 `torch.ops` 注册，因此实际
匹配结果是 `cann_bench.exp`。

## 5. CANN Bench 调用用户函数的位置

不采集 profiler 时，`src/kernel_eval/eval/op_runner.py` 的核心路径是：

```python
device_tensors = self.device_manager.to_device_batch(input_tensors)
updated_params = self._update_params(params, device_tensors)

with capture_output() as (cap_out, cap_err):
    outputs = func(**updated_params)
    self.device_manager.synchronize()
```

外层还有 `TorchOpGuard` 和 `DeviceResidencyGuard`，分别限制调用被禁止的 PyTorch
内置计算和把数据搬回 CPU 规避评测。对于 `Exp`，`func` 就是
`cann_bench.exp`，所以等价于：

```python
outputs = cann_bench.exp(
    x=npu_tensor,
    base=base,
    scale=scale,
    shift=shift,
)
```

## 6. JIT 到底发生在哪里

### 6.1 我们的触发代码

`examples/triton_ascend_cann_example/cann_bench/exp.py` 中这一句触发 launch：

```python
_exp_kernel[grid](
    x,
    output,
    n_elements,
    scale,
    shift,
    math.log(base) if base > 0 else 0.0,
    HAS_BASE=base > 0,
    BLOCK_SIZE=_BLOCK_SIZE,
)
```

### 6.2 Triton-Ascend 中 `[]` 做了什么

下面代码来自已验证环境安装的
`.venv/lib/python3.12/site-packages/triton/runtime/jit.py`，对应
`triton-ascend==3.2.1`、`triton==3.2.0`：

```python
class KernelInterface(Generic[T]):
    run: T

    def __getitem__(self, grid) -> T:
        return lambda *args, **kwargs: self.run(
            grid=grid, warmup=False, *args, **kwargs
        )
```

因此 `_exp_kernel[grid]` 先返回一个记录了 grid 的闭包；后面的 `(...)` 才调用
`JITFunction.run()`。

### 6.3 进程内缓存和编译分支

同一文件中，`JITFunction.run()` 的核心逻辑是：

```python
bound_args, sig_and_spec, constexpr_vals, non_constexpr_vals, excess_kwargs = \
    self.binder(*args, **kwargs)

key = ''.join(sig_and_spec) + str((constexpr_vals, excess_kwargs))
kernel = self.cache[device].get(key, None)

if kernel is None:
    kernel = self._do_compile(
        key, signature, device, backend, target,
        constants, options, configs[0], warmup,
    )

kernel.run(
    grid_0, grid_1, grid_2, stream,
    kernel.function, kernel.packed_metadata, launch_metadata,
    self.CompiledKernel.launch_enter_hook,
    self.CompiledKernel.launch_exit_hook,
    *non_constexpr_vals,
)
```

这里的第一层是进程内缓存 `self.cache[device]`。本示例至少会按以下信息产生不同
specialization：tensor 类型/签名、Triton 自动 specialization 信息、
`HAS_BASE`、`BLOCK_SIZE` 以及编译 options。`scale`、`shift` 和 `log_base` 是普通
运行时标量，不是 `tl.constexpr`，改变它们本身不会产生新的 kernel 源码。

### 6.4 磁盘缓存和 Ascend 编译流水线

进程内未命中后，`triton.compiler.compile()` 还会查询磁盘缓存。磁盘 key 包含
Triton 版本/实现、kernel AST/source、签名/constants、backend、编译 options 和会
使缓存失效的环境变量。命中时直接构造 `CompiledKernel`；未命中时由 Ascend backend
增加以下 stage：

```python
def add_stages(self, stages, options):
    if self.target.backend == "npu":
        stages["ttir"] = lambda src, metadata: make_ttir(
            src, metadata, options
        )
        stages["ttadapter"] = lambda src, metadata: ttir_to_linalg(
            src, metadata, options, named_ops=True
        )
        stages["npubin"] = lambda src, metadata: \
            linalg_to_bin_enable_npu_compile_910_95(
                src, metadata, options
            )
```

不同设备/模式可能选择另一条 `npubin` 分支或直接从 TTIR 编译，但当前已跑通的
Exp 缓存中真实生成了：

```text
_exp_kernel.ttir
_exp_kernel.ttadapter
_exp_kernel.npubin
_exp_kernel.json
__grp___exp_kernel.json
```

Docker 中固定：

```dockerfile
ENV TRITON_CACHE_DIR=/tmp/cann-bench-triton-cache
RUN mkdir -p ${TRITON_CACHE_DIR} && chmod 1777 ${TRITON_CACHE_DIR}
```

若没有设置 `TRITON_CACHE_DIR`，该版本默认使用
`$HOME/.triton/cache`。磁盘缓存让新进程也有机会复用已编译 binary；进程内缓存只在
当前 Python 进程有效。

## 7. 为什么 profiler 不统计首次 JIT 时间

性能路径复用 `PerfEvaluator`。它在进入 `torch_npu.profiler` 上下文之前无条件执行
一次用户函数并同步：

```python
# profiler 外：对于冷 specialization，这次调用完成 JIT
fn()
self._synchronize_profile_step()

with torch_npu.profiler.profile(
    schedule=torch_npu.profiler.schedule(
        wait=0, warmup=warmup, active=repeat, repeat=1
    ),
    ...,
) as prof:
    for i in range(warmup + repeat):
        fn_exc = self._run_profile_step(fn, prof)
```

默认 `warmup=3`、`repeat=5`。因此同一 case、同一 specialization 的首次 JIT 位于
profiler 之外，报告使用 profiler 中 NPU kernel 的设备时间，而不是 Python 首次调用
的墙钟时间。

边界条件：如果 profiler 迭代期间输入 dtype、shape-dependent constexpr 或编译
options 发生变化，仍可能出现新的 specialization。当前每个 case 的输入规格固定；
未来启用异构 InputPool 时需要额外约束或预热所有 specialization。

## 8. 我们提交示例的完整 Exp 核心源码

下面是实际构建进 wheel 并在 NPU 跑通的 `exp.py`：

```python
"""Triton-Ascend implementation of the CANN Bench Exp interface."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


_BLOCK_SIZE = 4096


@triton.jit
def _exp_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    scale,
    shift,
    log_base,
    HAS_BASE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    value = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    value = value * scale + shift
    if HAS_BASE:
        value = value * log_base
    tl.store(output_ptr + offsets, tl.exp(value), mask=mask)


def exp(
    x: torch.Tensor,
    base: float = -1.0,
    scale: float = 1.0,
    shift: float = 0.0,
) -> torch.Tensor:
    """Return ``exp((x * scale + shift) * log(base))`` on an NPU tensor.

    ``base <= 0`` selects the natural-base form used by the task schema. The
    wrapper only prepares metadata and output storage; the numerical work runs
    in ``_exp_kernel``.
    """
    if not x.is_contiguous():
        x = x.contiguous()

    output = torch.empty_like(x)
    n_elements = x.numel()
    if n_elements == 0:
        return output

    grid = (triton.cdiv(n_elements, _BLOCK_SIZE),)
    _exp_kernel[grid](
        x,
        output,
        n_elements,
        scale,
        shift,
        math.log(base) if base > 0 else 0.0,
        HAS_BASE=base > 0,
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    return output
```

设计边界：wrapper 负责连续布局、输出分配、元素数和 grid；`load`、公式计算、`exp`
和 `store` 都在 Triton kernel 中执行。`base > 0` 作为 constexpr，使有底数和自然底数
两条路径在编译期分支，而不是在每个元素上动态判断。

## 9. 构建代码

`build.sh` 只创建 wheel：

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

rm -rf build dist cann_bench.egg-info
python setup.py bdist_wheel

test -n "$(find dist -maxdepth 1 -type f -name 'cann_bench*.whl' -print -quit)"
```

`setup.py` 不声明平台依赖：

```python
from setuptools import find_packages, setup


setup(
    name="cann_bench",
    version="1.0.0",
    description="Triton Ascend operators for CANN Bench",
    packages=find_packages(),
    python_requires=">=3.10",
)
```

本次实际 wheel：

```text
cann_bench/__init__.py
cann_bench/exp.py
cann_bench/masked_scale.py
cann_bench/mish.py
cann_bench/sigmoid.py
cann_bench/swi_glu.py
cann_bench-1.0.0.dist-info/{METADATA,WHEEL,top_level.txt,RECORD}
```

SHA256：

```text
32838d2eae06b513f2b88e9da40bf712d6882a39807dd875cf58116614b4ce14
```

该摘要对应 2026-07-23 的验证产物；当前 wheel 构建包含文件时间戳，不应把该摘要
当作跨机器可复现构建标识。

用户可以提交源码目录，让 CANN Bench 调用其中的 `build.sh`；也可以直接提交构建出的
wheel。示例构建命令：

```bash
cd examples/triton_ascend_cann_example
bash build.sh
unzip -l dist/cann_bench-1.0.0-py3-none-any.whl
```

先只跑正确性：

```bash
./scripts/run_evaluation.sh \
  --source-dir examples/triton_ascend_cann_example \
  --task-dir tasks/level1/exp \
  --operator Exp \
  --device-id 0 \
  --no-perf
```

正确性通过后去掉 `--no-perf`，进入 pre-flight、JIT cache 和 CANN profiler 路径：

```bash
./scripts/run_evaluation.sh \
  --source-dir examples/triton_ascend_cann_example \
  --task-dir tasks/level1/exp \
  --operator Exp \
  --device-id 0
```

## 10. 我们改了哪些代码

| 文件 | 作用 |
|---|---|
| `requirements-triton.txt` | 固定可选 Triton-Ascend 运行时入口 |
| `docker/Dockerfile` | 增加可选 Triton-Ascend flavor 和可写 JIT cache |
| `docker/test_env.py` | 不只 import，真实 JIT 并执行 NPU vector add |
| `src/kernel_eval/data/package_manager.py` | 同版本 wheel 强制重装，清理全部 submission 子模块缓存 |
| `examples/triton_ascend_cann_example/**` | 可提交的五算子 Triton wheel 示例 |
| `tests/ut/test_triton_ascend_example.py` | wheel 结构、launch 参数和可选真机 NPU 测试 |

`OperatorMatcher`、`OpRunner`、精度比较和 CANN profiler 原本已经面向普通 callable，
本次直接复用，没有增加 Triton 专用分支。这是集成方案的核心：DSL 差异留在用户
wrapper 和平台运行时中，评测生命周期保持统一。

## 11. 怎么知道用户算子包里有什么

wheel 是 ZIP。安装前可以枚举归档、读取 `METADATA`/`RECORD`、检查 `.py`/`.so`
类型；安装后可以扫描公开 callable 和函数签名。两层信息不同：

| 层次 | 能回答的问题 | 当前状态 |
|---|---|---|
| wheel 静态内容 | 有哪些文件、包名、是否携带二进制 | 本次已人工检查；框架尚未做完整强制审计 |
| import 后接口 | 导出了哪些 callable、签名是什么 | `scan_interfaces()` 已实现 |
| 运行行为 | 是否调用违规 API、是否把数据搬回 CPU | Torch/Device Guard 已实现 |

必须明确：`import cann_bench` 会执行用户顶层 Python 代码，所以“import 后扫描”不是
安全沙箱。生产评测还应在 import 前增加：

1. 拒绝路径穿越、symlink、未知顶层包和非白名单文件类型；
2. 默认拒绝 submission 自带 `.so`、可执行文件和依赖包，按赛题类型例外放行；
3. 校验 wheel `RECORD`，记录整个提交 SHA256；
4. 对 Python AST/import 做策略检查；
5. 在非 root、无网络、资源受限的独立容器里执行 import 和评测。

`--no-deps`、TorchOpGuard 和 DeviceResidencyGuard 是必要控制，但不能替代 submission
容器隔离和 import 前审计。

## 12. 环境、验证与证据

本次真机验证环境实际读数：

```text
Python          3.12.9
torch           2.7.1+cpu
torch_npu       2.7.1.post6
triton-ascend   3.2.1
triton          3.2.0
backend         npu
target          Ascend950PR_9579
device count    1
```

执行命令：

```bash
PYTHONPATH=src CANN_BENCH_RUN_TRITON_NPU=1 \
  .venv/bin/python -m pytest -q tests/ut/test_triton_ascend_example.py
```

结果为 `10 passed`：5 个结构/调用合同测试，加上 Exp、Sigmoid、Mish、MaskedScale、
SwiGLU 5 个真实 NPU 执行测试。

另有 CANN profiler 归档证明 `_exp_kernel` 被识别为 `AI_VECTOR_CORE` kernel；
`level1/exp_1` 精度误差为 0，五次设备时间的中位数为 `6.71 us`。这证明执行的不是
Python 模拟函数，而是已生成并 launch 的 NPU kernel。

仓库 Docker 默认目标是 CANN 9.0.0，但 Triton-Ascend 与 CANN/驱动/设备的兼容性
不能只靠 import 或版本字符串判断。每个正式镜像必须以 `docker/test_env.py` 的真实
vector-add JIT/NPU smoke 作为准入条件。

## 13. 是否可以修改 Triton-Ascend 优化

可以，但应按层次决定修改位置：

1. 单个算子的 grid、block size、访存和地址计算问题，优先修改 submission。
2. 多个算子共同出现相同的 lowering、vector math 或冗余 cast 问题，再修改
   Triton-Ascend backend。
3. backend 修改后构建新的平台 runtime wheel，放入 Docker flavor；用户的
   `cann_bench` submission 合同不变。
4. 每次 backend 改动保存 TTIR、ttadapter、npubin 和 profiler 做 A/B，并回归五个
   算子，避免只优化一个 kernel 却造成通用 codegen 回退。

当前已经有一个典型 submission 级优化：SwiGLU 从逐元素除法/取模寻址改为二维 grid，
case 1 从 `642.72 us` 降到 `21.16 us`。这类问题不需要改编译器。只有 Exp、Sigmoid、
Mish 等多个数学算子都显示相同 lowering 瓶颈时，才值得进入 Triton-Ascend backend。

## 14. 关键源码位置

- submission：`examples/triton_ascend_cann_example/cann_bench/exp.py`
- 构建：`examples/triton_ascend_cann_example/build.sh`
- 安装与扫描：`src/kernel_eval/data/package_manager.py`
- 名称匹配：`src/kernel_eval/benches/cann_matcher.py`
- callable 执行：`src/kernel_eval/eval/op_runner.py`
- JIT 预检与 profiler：`src/kernel_eval/eval/perf_eval.py`
- 运行时 smoke：`docker/test_env.py`
- Triton JIT 入口：`.venv/lib/python3.12/site-packages/triton/runtime/jit.py`
- Triton 编译/磁盘 cache：`.venv/lib/python3.12/site-packages/triton/compiler/compiler.py`
- Ascend backend：`.venv/lib/python3.12/site-packages/triton/backends/ascend/compiler.py`
