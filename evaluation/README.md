# 评测流水线

运行一次提交所需的全部组件，以及用于打包 benchmark 和在本地复现一次评测的辅助工具。

## 调用关系

```
                   ┌─────────────────────────────────────────────┐
  cann-bench       │ runner/main.py                              │
  控制面     ──────▶│  阶段: prepare → compile → correctness      │
                   │        → performance → archive              │
                   └──────┬──────────────────────────────────────┘
                          │ 每个阶段执行一个 shell 脚本
                          ▼
                    harness/*.sh   (runner 契约 — 每阶段入口)
                          │
                          ▼ (run_correctness.sh / run_perf.sh)
                      evaluate.py   (总调度器)
                          │
                          ▼
                       core/*.py   (内部模块: 精度 / 数据 / 参数)
```

## 目录结构

| 路径 | 角色 |
| --- | --- |
| `evaluate.py` | 主调度器。安装提交 wheel，加载 `proto.yaml` / `cases.yaml` / `golden.py`，执行正确性校验和（可选的）性能测量，写出 `evaluation_results.json`。 |
| `harness/` | 由 runner 调用的各阶段 shell 脚本（在每个 bundle 的 `benchmark.yaml` 中引用）。`build.sh` 负责 wheel 构建；`run_correctness.sh` / `run_perf.sh` 配置 `ASCEND_CUSTOM_OPP_PATH` 并调用 `evaluate.py`；`_common.sh` 汇总它们共享的初始化逻辑。 |
| `core/` | `evaluate.py` 依赖的 Python 模块 — `case_loader`、`data_generator`、`dtype_mapper`、`param_builder`、`precision_checker`、`profiler_manager`。没有其它消费者。 |
| `tools/` | 开发者 / 管理员辅助工具，**不在** runner 的关键路径上。`register_benchmark.py` 将一个算子打包成 `.tar.gz` bundle（并可选上传）；`simulate_runner.sh` 在本地 NPU 上端到端复现一次 job；`summarize.py` 将 `evaluation_results.json` + `result.json` 渲染成易读的 `summary.md`。 |
| `submission_examples/` | 三份参考提交 (`aclnn_launch_example`、`direct_launch_example`、`direct_launch_simple_example`)，会构建出评测器所需形状的 `cann_bench` wheel。 |
| `result_examples/` | 针对上述示例提交运行模拟器得到的样例输出。详见 `result_examples/README.md`。 |

## 性能测量方式

`evaluate.py` 的 `measure_perf` 通过 `torch_npu.profiler.profile` 采集 NPU 端的 chrome trace，解析出设备内核事件的累计耗时。与使用 `torch.npu.Event` 测壁钟时间相比，profiler 方式能剥离 Python/OpCommand 派发开销，只保留 kernel 真正执行的微秒数——这样 aclnn 风格与 direct-launch 风格的提交可以在同一把尺子下对比。

每次 profile 前会先执行一对 MatMul + ReduceMax 用于 NPU 升频、清 L2 cache；解析 trace 时再把这两个 kernel 过滤掉，保证只统计目标算子。实现位于 `core/profiler_manager.py`。

## 精度参考

- golden 在 **CPU fp64** 下计算，比 NPU 原生 dtype 精度更高，避免溢出/下溢同时污染参考值。
- 比较阶段 (`core/precision_checker.py`) 将双方 cast 回 fp32 后计算 MERE / MARE，阈值按 `cases.yaml` 声明的 NPU dtype 选取（fp16: 2⁻¹⁰、fp32: 2⁻¹³、bf16: 2⁻⁷），满足 MERE < 阈值且 MARE < 10× 阈值即为通过。

## Baseline

速度对比的分母 `baseline_perf_us` 默认取 `cases.yaml` 的值（支持 `910b2` 默认值或字典形式的多硬件展开）。对命中硬件无 baseline 的 case，`pre_measure_baselines` 会在 **安装 submission wheel 之前** 用相同的 profiler 路径测一次 golden 作为 baseline。

带上 `--measure-baselines`（或在 harness 中设置 `BENCH_MEASURE_BASELINES=1`）会重测每一条 case 的 golden，把测得值作为 baseline，同时在输出 JSON 中同时记录 `baseline_yaml_us` 和 `baseline_measured_us`，方便对照、校准 yaml。

## 本地跑一次 job

需要一台可以访问 NPU 的机器，装有 `torch_npu` 且能找到 `ASCEND_HOME_PATH`。示例如下：

```bash
# 1. 构建示例提交的 wheel
bash evaluation/submission_examples/direct_launch_simple_example/build.sh

# 2. 在本地 NPU 上模拟完整 job
bash evaluation/tools/simulate_runner.sh \
    evaluation/submission_examples/direct_launch_simple_example/dist/*.whl \
    /path/to/bundle \
    direct_launch_simple
```

bundle 的目录布局与 `tools/register_benchmark.py` 产出的格式一致，详见该脚本。
