# 快速入门

**文档版本：V0.2.0**

本文档介绍如何使用评测工程进行算子代码生成评测。

## 前置条件

- Python 3.8+
- PyTorch 2.0+
- torch_npu（NPU 模式）
- CANN 环境（NPU 模式）

## 安装

仓内运行不需要 `pip install`——只需安装依赖并配置 `PYTHONPATH`：

```bash
pip install -r requirements.txt
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
```

> 当前仓库未提供 `pyproject.toml` / `setup.py`，所以 `pip install -e .` 会失败。
> 若需打包发布或独立安装，请先在仓库根目录补 `pyproject.toml`。

## 评测命令

### 从源码目录评测（推荐）

自动扫描、编译、安装 AI 生成的算子源码：

```bash
./scripts/run_evaluation.sh --source-dir /path/to/ai_ops
```

### 评测指定算子

```bash
# 评测指定目录
./scripts/run_evaluation.sh --task-dir tasks/level1

# 评测单个算子目录
./scripts/run_evaluation.sh --task-dir tasks/level1/exp

# 按算子名称筛选
./scripts/run_evaluation.sh --operator Exp

# 评测单个用例
./scripts/run_evaluation.sh --operator Exp --case-id 1

# CPU 模式评测
./scripts/run_evaluation.sh --device cpu --operator Exp

# 设置 warmup/repeat 参数
./scripts/run_evaluation.sh --operator Exp --warmup 5 --repeat 10
```

### 多卡并行评测

不指定 `--device-id` 时自动使用全部可用 NPU 卡：

```bash
# 多卡并行（自动检测）
./scripts/run_evaluation.sh --operator Exp

# 指定每卡进程数
./scripts/run_evaluation.sh --operator Exp --processes-per-card 4

# 指定进程超时
./scripts/run_evaluation.sh --operator Exp --timeout-per-process 600
```

### 单卡评测

```bash
# 单卡模式（指定设备 ID）
./scripts/run_evaluation.sh --device-id 0 --operator Exp
```

### 查看算子信息

```bash
# 列出所有算子
./scripts/run_evaluation.sh -a list

# 查看算子详情
./scripts/run_evaluation.sh -a info --operator Exp

# 查看配置
./scripts/run_evaluation.sh -a config
```

## 高级选项

`./scripts/run_evaluation.sh` 支持的参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--device <type>` | 设备类型 (cpu/npu) | npu |
| `--device-id <id>` | NPU 设备 ID（不指定则多卡并行） | None |
| `--processes-per-card <n>` | 每卡进程数（多卡模式） | 2 |
| `--timeout-per-process <n>` | 单进程超时（秒，等价于 cli `--timeout-per-operator`） | 300 |
| `--warmup <n>` | 预热次数 | 3 |
| `--repeat <n>` | 采集次数 | 5 |
| `--no-perf` | 关闭性能采集（仅精度验证） | False |
| `--profiler-level <level>` | Profiler 级别 (Level1/Level2) | Level1 |

> shell 暴露的是最常用子集。若需要 `--no-subprocess-isolation` / `--op-timeout-sec` / `--no-iterative-compile` / `--reports-dir` / `--eval-code` 等更精细控制，请直调 cli：
> ```bash
> PYTHONPATH=src python -m kernel_eval.cli eval --source-dir /path/to/ai_ops \
>     --no-subprocess-isolation --op-timeout-sec 480
> ```
> cli 完整参数表见 [evaluator_design.md §3.3](../design/evaluator_design.md#33-命令行参数)。

## 测试脚本

使用 `tests/run_simple.py` 进行 Golden 验证：

```bash
# CPU 简单验证
./scripts/run_test.sh --cpu --operator Exp

# NPU 模拟评测（Golden 伪装成 AI 算子）
./scripts/run_test.sh --npu --operator Exp

# 指定设备 ID
./scripts/run_test.sh --npu --device-id 1 --operator Exp

# 多卡并行
./scripts/run_test.sh --npu --operator Exp
```

## 评测报告

评测完成后，报告输出到 `reports/` 目录：

- `reports/eval_report.json`：JSON 格式详细报告
- `reports/eval_report.md`：Markdown 格式报告
- `reports/summary.md`：摘要报告
- `reports/prof_data/`：性能采集数据

## 跑测试：inner/tests/

整合后所有测试集中在 `inner/tests/`：

```bash
# task 一致性（CPU-only，ubuntu-latest CI 必跑）
uv run python -m pytest inner/tests/task/ -v

# baseline-bench（需要 NPU 自托管 runner，CANN 9.0.0）
CANN_BENCH_DEVICE=0 uv run python -m pytest inner/tests/baseline/ -v

# 把 NPU 测试结果回填 cases.yaml + cases.csv
# --dry-run 看 diff（不写盘）
uv run python inner/tests/apply_baselines.py --dry-run \
    --input inner/tests/baseline/results/baseline_perf_*.json
# --apply 默认只填 null 项 (保护人工录入); rebaseline 已有 baseline 必加 --force
uv run python inner/tests/apply_baselines.py --apply --force \
    --input inner/tests/baseline/results/baseline_perf_*.json

# Drift gate (CI 上 self-hosted 跑)
uv run python inner/tests/apply_baselines.py --check --tolerance 0.20 \
    --input inner/tests/baseline/results/baseline_perf_*.json
```

详见 [部署文档](self_hosted_runner_cann9.md) 关于 NPU runner 容器化。

## 下一步

- [贡献指南](contributing.md)：如何提交新算子评测任务
- [评测基准规范](../spec/benchmark_spec.md)：算子定义和精度标准
- [评测工程设计](../design/evaluator_design.md)：评测器架构设计
- [性能采集设计](../design/perf_collection_design.md)：性能采集机制设计