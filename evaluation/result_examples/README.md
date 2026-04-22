# 示例运行结果

由 runner（`runner/main.py` → `harness/*.sh` → `evaluate.py`）在 NPU 主机上运行后输出的样例。每个子目录就是一次 job 写入 `BENCH_OUTPUT_DIR` 的内容：

```
<submission_type>/
  evaluation_results.json   # 各算子的精度与加速比，上传给控制面
  result.json               # 各阶段耗时，由 runner.main.build_result() 生成
  summary.md                # 两份 JSON 的人类可读汇总（由 summarize.py 渲染）
  logs/
    compile.log             # harness/build.sh 的 stdout/stderr
    correctness.log         # harness/run_correctness.sh 的 stdout/stderr
    performance.log         # harness/run_perf.sh 的 stdout/stderr
```

## 如何重新生成

在仓库根目录执行：

```bash
# 1. 构建 wheel（只需一次）
bash evaluation/submission_examples/direct_launch_simple_example/build.sh
bash evaluation/submission_examples/aclnn_launch_example/build.sh --soc=ascend910b

# 2. 安装 aclnn 的 .run 包，让 aclnn 算子在运行时可见
"evaluation/submission_examples/aclnn_launch_example/dist/cann_bench_*.run" \
  --install-path=/usr/local/Ascend/cann-8.5.0/opp --quiet

# 3. 构造一个 bundle（单算子或多算子批量），目录结构与 tools/register_benchmark.py
#    产出的格式一致：
#      benchmark.yaml  harness/  evaluate.py  evaluation/core/  data/...

# 4. 运行模拟器（与 runner/main.py 使用同一套环境变量契约）
bash evaluation/tools/simulate_runner.sh \
  evaluation/submission_examples/direct_launch_simple_example/dist/cann_bench_ops-1.0.0-cp38-abi3-linux_aarch64.whl \
  /path/to/bundle \
  direct_launch_simple
```

校准 yaml baseline 时可开启 `--measure-baselines`：

```bash
BENCH_MEASURE_BASELINES=1 bash evaluation/tools/simulate_runner.sh \
  <wheel> <bundle> calibration_run
```

输出的 `summary.md` 会在 baseline 列并列展示 yaml 值与实际 NPU 测得值，便于对照更新 `cases.yaml`。

## evaluation_results.json 数据结构

```
{
  "hardware": "910b2",
  "total_operators": 3,
  "total_cases":     36,
  "total_passed":    <int>,
  "overall_geometric_mean_speedup": <float>,
  "operators": [
    {
      "operator":   "Add",
      "hardware":   "910b2",
      "total_cases":      8,
      "passed_cases":     <int>,
      "geometric_mean_speedup": <float>,
      "results": [
        {
          "case_id":             1,
          "case_name":           "float32-1D-1K",
          "status":              "PASS" | "FAIL",
          "detail":              "MERE=0.00e+00 MARE=0.00e+00",
          "mere":                <float>,       # 平均相对误差
          "mare":                <float>,       # 最大相对误差
          "speedup":             <float>|null,
          "baseline_perf_us":    <float>,       # 实际用于计算 speedup 的 baseline
          "baseline_yaml_us":    <float>|null,  # cases.yaml 里写的那个
          "baseline_measured_us":<float>|null,  # 本次在 NPU 上重测得到的 golden 时间
          "custom_time_us":      <float>|null
        }, ...
      ]
    }, ...
  ]
}
```

`hardware` 通过 `evaluate.py --hardware <name>` 选定（默认 `910b2`）。当 `baseline_perf_us` 是字典时，评测器按所选硬件查对应 key；当它是标量时，仅在运行在默认硬件（`910b2`）上时有意义。
