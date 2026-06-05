# Scripts

本目录包含评测工程运行脚本和工具脚本。

**目录结构**：

```
scripts/
├── run_evaluation.sh        # AI算子源码评测（完整流程：编译→安装→评测）
├── run_auto_pipeline.sh     # auto_pipeline 的 python -m 包装入口
├── run_ut.sh                # 单元测试运行
├── utils/
│   ├── build_golden_wheel.sh # Golden whl 打包（Shell 入口）
│   ├── build_golden_wheel.py # Golden whl 打包（Python 逻辑）
│   ├── yaml_to_csv.py       # YAML → CSV 转换
│   └── yaml_block_to_flow.py  # YAML block → flow 格式转换
└── README.md
```

---

## run_auto_pipeline.sh

auto_pipeline 的 shell 包装，等价于在仓库根目录设置 `PYTHONPATH=src` 后执行
`python -m auto_pipeline.cli`。

### 用法

```bash
export PTO_TILE_LIB_CODE_PATH=/path/to/pto-isa
./scripts/run_auto_pipeline.sh \
  --config path/to/config.yaml \
  --workspace /path/to/agent/repo \
  --model deepseek/deepseek-v4-pro \
  --devices 0 \
  --parallel 1
```

`run_auto_pipeline.sh` 只设置 `PYTHONPATH=src` 并转发到 `python -m auto_pipeline.cli`。
当前 `auto_pipeline` 的使用说明见
[`docs/guide/auto_pipeline_usage.md`](../docs/guide/auto_pipeline_usage.md)，实现设计见
[`docs/design/benchmark_orchestrator_architecture.md`](../docs/design/benchmark_orchestrator_architecture.md)。

---

## run_evaluation.sh

AI 生成算子源码评测脚本，支持完整流程（扫描→编译→安装→评测）。

### 用法

```bash
./scripts/run_evaluation.sh [源码目录] [选项]
```

### 命令行选项

#### 操作选项

| 选项 | 说明 |
|------|------|
| `-a, --action <action>` | 操作类型：`eval`(评测)、`list`(列表)、`info`(详情)、`config`(配置)，默认：`eval` |
| `--source-dir <dir>` | AI 生成的算子源码目录（自动扫描编译安装） |

#### 目录配置

| 选项 | 说明 |
|------|------|
| `--task-dir <path>` | 指定评测目录，默认：`tasks`。支持：`tasks`、`tasks/level1`、`tasks/level1/exp` 等 |

#### 设备配置

| 选项 | 说明 |
|------|------|
| `--device <type>` | 设备类型：`cpu`、`npu`（默认：`npu`） |
| `--device-id <id>` | 指定 NPU 设备 ID（单卡模式）。不指定则自动使用全部可用卡（多卡并行） |

#### 多进程并行配置

| 选项 | 说明 |
|------|------|
| `--processes-per-card <n>` | 每卡进程数（默认：2） |
| `--timeout-per-process <n>` | 单进程超时（秒，默认：300）。**等价于 cli 的 `--timeout-per-operator`**；shell 通过环境变量 `KERNEL_BENCH_TIMEOUT_PER_PROCESS` 传给底层 |

#### 用例筛选

| 选项 | 说明 |
|------|------|
| `--operator <name>` | 按算子名称筛选 |
| `--case-id <id>` | 按用例编号筛选 |

#### 性能配置

| 选项 | 说明 |
|------|------|
| `--warmup <n>` | 预热次数（默认：3） |
| `--repeat <n>` | 采集次数（默认：5） |
| `--no-perf` | 关闭性能采集，仅做精度验证 |
| `--profiler-level <level>` | Profiler 级别：`Level1`、`Level2`（默认：`Level1`） |

#### 其他选项

| 选项 | 说明 |
|------|------|
| `-v, --verbose` | 详细输出 |
| `-h, --help` | 显示帮助信息 |

> shell 只透传最常用参数子集。若需要 `--no-subprocess-isolation` / `--op-timeout-sec` / `--no-iterative-compile` / `--reports-dir` / `--eval-code` 等，请直调 cli（`python -m kernel_eval.cli eval ...`），完整参数表见 [docs/design/evaluator_design.md §3.3](../docs/design/evaluator_design.md#33-命令行参数)。

### 使用示例

#### 从源码目录评测（推荐）

```bash
# 位置参数方式
./scripts/run_evaluation.sh /path/to/ai_ops

# 显式指定 --source-dir
./scripts/run_evaluation.sh --source-dir /path/to/ai_ops

# 评测指定算子
./scripts/run_evaluation.sh /path/to/ai_ops --operator Exp

# 评测单个用例
./scripts/run_evaluation.sh /path/to/ai_ops --operator Exp --case-id 1
```

#### 指定评测目录

```bash
# 评测 tasks 目录（默认）
./scripts/run_evaluation.sh

# 评测指定级别目录
./scripts/run_evaluation.sh --task-dir tasks/level1

# 评测单个算子目录
./scripts/run_evaluation.sh --task-dir tasks/level1/exp

# 按算子名称筛选
./scripts/run_evaluation.sh --operator Exp
```

#### 设备配置

```bash
# CPU 评测
./scripts/run_evaluation.sh --device cpu --operator Exp

# 单卡 NPU 评测
./scripts/run_evaluation.sh --device-id 0 --operator Exp

# 多卡并行评测（自动检测全部可用卡）
./scripts/run_evaluation.sh --operator Exp
```

#### 性能配置

```bash
# 仅精度验证（关闭性能采集）
./scripts/run_evaluation.sh --no-perf --operator Exp

# 自定义预热和采集次数
./scripts/run_evaluation.sh --warmup 5 --repeat 10 --operator Exp

# 使用 Level2 Profiler（更详细）
./scripts/run_evaluation.sh --profiler-level Level2 --operator Exp
```

#### 查看算子信息

```bash
# 列出算子
./scripts/run_evaluation.sh -a list

# 查看算子详情
./scripts/run_evaluation.sh -a info --operator Exp

# 查看配置
./scripts/run_evaluation.sh -a config
```

### 输出

评测报告保存在 `reports/` 目录：
- `reports/eval_report.json`：JSON 格式详细报告
- `reports/eval_report.md`：Markdown 格式报告
- `reports/summary.md`：摘要报告

---

## build_golden_wheel.sh

Golden whl 打包脚本，将 tasks/ 下 golden.py 函数收集打包成纯 Python `cann_bench` whl 包，
注册到 `torch.ops.cann_bench` 命名空间，用于验证 golden(NPU) 与 golden(CPU fp64) 的精度一致性。

### 用法

```bash
# 构建所有算子并安装
./scripts/utils/build_golden_wheel.sh --install

# 只打包指定算子
./scripts/utils/build_golden_wheel.sh --operator Mish --install

# 只打包指定级别
./scripts/utils/build_golden_wheel.sh --level 1 --install

# 构建后通过 run_evaluation 评测
./scripts/run_evaluation.sh --task-dir tasks/level1/mish --no-perf
```

详见 `scripts/utils/build_golden_wheel.py` 中的完整参数说明。

---

## run_ut.sh

单元测试运行脚本，基于 pytest。

### 用法

```bash
./scripts/run_ut.sh [选项]
```

### 命令行选项

| 选项 | 说明 |
|------|------|
| `-v, --verbose` | 详细模式，显示每个测试名称和结果 |
| `-q, --quiet` | 静默模式，只显示最终统计 |
| `-k, --keyword <kw>` | 按关键字筛选测试（pytest -k） |
| `-f, --file <name>` | 指定测试文件（如 test_config.py） |
| `-t, --test <spec>` | 指定测试方法（如 TestConfig::test_default_config） |
| `-x, --fail-fast` | 首次失败即停止 |
| `-s, --no-capture` | 不捕获 stdout/stderr（调试用） |
| `-j, --jobs <n>` | 并行执行（pytest-xdist） |
| `--pdb` | 失败时进入 pdb 调试器 |
| `-h, --help` | 显示帮助 |

### 使用示例

```bash
# 全部单元测试
./scripts/run_ut.sh

# 详细模式
./scripts/run_ut.sh -v

# 按关键字筛选
./scripts/run_ut.sh -k "config"

# 指定测试文件
./scripts/run_ut.sh -f test_config.py

# 指定测试方法
./scripts/run_ut.sh -f test_config.py -t TestConfig::test_default_config

# 失败即停 + 详细输出
./scripts/run_ut.sh -x -v

# 无捕获 + 失败调试
./scripts/run_ut.sh -s --pdb
```

---

## utils/yaml_to_csv.py

将 cases.yaml 转换为 cases.csv 格式。

### 用法

```bash
python scripts/utils/yaml_to_csv.py <input.yaml> [-o <output.csv>]
```

### 命令行选项

| 选项 | 说明 |
|------|------|
| `input_file` | 输入 YAML 文件路径 |
| `-o, --output <path>` | 输出 CSV 文件路径，默认与输入同名但扩展名改为 .csv |

### 使用示例

```bash
# 转换为同名 CSV
python scripts/utils/yaml_to_csv.py tasks/level1/exp/cases.yaml

# 指定输出路径
python scripts/utils/yaml_to_csv.py tasks/level1/exp/cases.yaml -o output.csv
```

---

## utils/yaml_block_to_flow.py

将 YAML 文件中的 block format 数组转换为 flow format。

### 用法

```bash
python scripts/utils/yaml_block_to_flow.py <input.yaml> [-o <output.yaml>] [-s <style>]
```

### 命令行选项

| 选项 | 说明 |
|------|------|
| `input_file` | 输入 YAML 文件路径 |
| `-o, --output <path>` | 输出 YAML 文件路径，默认输出到控制台 |
| `-i, --indent <n>` | 缩进空格数（默认：2） |
| `-s, --style <style>` | 转换风格：`all`（所有列表）、`selective`（仅简单列表）、`smart`（智能选择），默认：`selective` |

### 使用示例

```bash
# 转换为 flow format（输出到控制台）
python scripts/utils/yaml_block_to_flow.py cases.yaml

# 保存到文件
python scripts/utils/yaml_block_to_flow.py cases.yaml -o cases_flow.yaml

# 转换所有列表为 flow format
python scripts/utils/yaml_block_to_flow.py cases.yaml -s all

# 智能转换
python scripts/utils/yaml_block_to_flow.py cases.yaml -s smart
```

---

## 脚本对比

| 脚本 | 用途 | 底层入口 | 输入 | 输出 |
|------|------|----------|------|------|
| `run_evaluation.sh` | AI 算子完整评测 | `src/kernel_eval/cli.py` | AI 源码目录 / tasks 目录 | 评测报告 |
| `run_test.sh` | 测试（ut/e2e） | pytest | tests/ut + tests/e2e 目录 | pytest 输出 |
| `utils/build_golden_wheel.sh` | Golden whl 打包 | `scripts/utils/build_golden_wheel.py` | tasks 目录 | cann_bench whl |
| `utils/gen_report.sh` | 评测报告生成 | `kernel_eval.report.html_generator` | eval JSON 文件 | HTML 报告 |

**底层入口职责**：

| 入口 | 职责 | 核心能力 |
|------|------|----------|
| `cli.py` | **评测核心** | `--task-dir`、多卡并行、`--no-perf`、`--source-dir`、`--profiler-level` |

**参数统一说明**：

`run_evaluation.sh` 支持以下参数：

| 参数 | 说明 |
|------|------|
| `--task-dir <path>` | 指定评测目录（替代原来的 `--level`） |
| `--device-id <id>` | 单卡模式：指定设备；不指定则多卡并行 |
| `--processes-per-card <n>` | 多卡并行时每卡进程数 |
| `--warmup <n>` | 预热次数 |
| `--repeat <n>` | 采集次数 |
| `--no-perf` | 关闭性能采集，仅精度验证 |
| `--operator <name>` | 按算子名称筛选 |
| `--profiler-level <level>` | Profiler 级别（Level1/Level2） |

---

## 相关文档

- [评测基准规范](../docs/spec/benchmark_spec.md)
- [评测工程设计](../docs/design/evaluator_design.md)
- [性能采集设计](../docs/design/perf_collection_design.md)
- [快速入门](../docs/guide/quick_start.md)
