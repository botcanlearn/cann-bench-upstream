# Test Scripts

本目录包含运行测试和维护测试数据的脚本。

## 目录结构

```
scripts/
├── run_test.sh            # 统一的测试运行脚本
├── update_baseline_perf.py # 更新基线性能数据脚本
└── README.md              # 本文档
```

## 快速开始

```bash
# 进入项目根目录
cd /path/to/cann-bench

# 运行所有测试
./scripts/run_test.sh

# 运行 CPU 测试
./scripts/run_test.sh --cpu

# 运行指定算子测试
./scripts/run_test.sh --operator gelu

# 运行 Level 1 测试
./scripts/run_test.sh --level 1

# 查看帮助
./scripts/run_test.sh --help
```

## 命令行选项

### 设备选项

| 选项 | 说明 |
|------|------|
| `--cpu` | 使用 CPU 设备测试（默认） |
| `--npu` | 使用 NPU 设备测试 |

### 筛选选项

| 选项 | 说明 | 示例 |
|------|------|------|
| `--level <1-4>` | 按难度级别筛选 | `--level 1` |
| `--operator <name>` | 按算子名称筛选（模糊匹配） | `--operator gelu` |
| `--case-id <num>` | 按用例编号筛选 | `--case-id 1` |

### 输出选项

| 选项 | 说明 |
|------|------|
| `-v, --verbose` | 详细输出（显示 shape、dtype 等） |
| `--prof` | 启用性能采集 |
| `-o, --output <path>` | 指定结果输出文件路径 |

## 使用示例

### 基本用法

```bash
# 运行所有测试
./scripts/run_test.sh

# CPU 设备测试（默认）
./scripts/run_test.sh --cpu

# NPU 设备测试
./scripts/run_test.sh --npu
```

### 按级别筛选

```bash
# Level 1 测试（基础算子）
./scripts/run_test.sh --level 1

# Level 2 测试
./scripts/run_test.sh --level 2

# Level 3 测试
./scripts/run_test.sh --level 3

# Level 4 测试（高难度融合算子）
./scripts/run_test.sh --level 4
```

### 按算子筛选

```bash
# 运行 gelu 算子测试
./scripts/run_test.sh --operator gelu

# 运行 cross_entropy_loss 测试
./scripts/run_test.sh --operator cross_entropy_loss

# 运行所有包含 matmul 的算子测试（模糊匹配）
./scripts/run_test.sh --operator matmul
```

### 组合选项

```bash
# CPU 设备 + Level 1 测试
./scripts/run_test.sh --cpu --level 1

# NPU 设备 + matmul 算子 + 详细输出
./scripts/run_test.sh --npu --operator matmul -v

# Level 2 + 指定算子 + 指定用例
./scripts/run_test.sh --level 2 --operator add --case-id 1

# NPU 测试 + 性能采集
./scripts/run_test.sh --npu --prof
```

## 测试报告

测试结果默认保存在 `test/reports/` 目录：

```bash
# 查看最新测试报告
cat test/reports/test_results.json

# 查看性能 trace（NPU 测试 + --prof）
ls test/reports/traces/
```

报告格式：

```json
{
  "summary": {
    "total": 100,
    "passed": 98,
    "failed": 2,
    "skipped": 0,
    "pass_rate": "98.00%",
    "timestamp": "2026-04-09T17:00:00"
  },
  "results": [
    {
      "level": 1,
      "operator": "Gelu",
      "case_id": 1,
      "status": "success",
      "elapsed_us": 3050.5,
      "device": "cpu"
    }
  ]
}
```

## 注意事项

1. 运行 NPU 测试前确保 NPU 设备可用
2. Level 4 测试可能需要较长执行时间，建议单独运行
3. 使用 `--prof` 参数会在 `reports/traces/` 生成性能 trace 文件

---

## update_baseline_perf.py

将测试结果中的性能数据更新到 cases yaml 文件的 `baseline_perf_us` 字段。

### 用法

```bash
python scripts/update_baseline_perf.py <test_results.json> <kernel_bench_dir>
```

### 示例

```bash
# 更新所有 case 的基线性能
python scripts/update_baseline_perf.py test/reports/test_results.json kernel_bench
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `test_results.json` | 测试结果 JSON 文件路径 |
| `kernel_bench_dir` | kernel_bench 目录路径 |

### 功能说明

1. 从 `test_results.json` 读取成功用例的 `elapsed_us` 性能数据
2. 将性能值保留两位小数（如 `33.78` -> `33.78`, `95.13999` -> `95.14`）
3. 根据 `level/operator/case_id` 定位到对应的 cases yaml 文件
4. 更新 yaml 文件中对应 case 的 `baseline_perf_us` 字段

### 输出示例

```
[INFO] 加载测试结果: test/reports/test_results.json
[INFO] 构建性能映射表...
[INFO] 共 659 个成功用例的性能数据

[INFO] 处理 level1...
  [1/swiglu] case 1: None -> 34.0
  [1/swiglu] case 2: None -> 95.0
  [1/gelu] case 1: None -> 23.0
  ...

[DONE] 共更新 659 个 case 的 baseline_perf_us
```

### 依赖

- `ruamel.yaml`：保留 yaml 原始格式（推荐）
  ```bash
  pip install ruamel.yaml
  ```

如未安装 `ruamel.yaml`，脚本会使用标准 `yaml` 库，但可能丢失 inline 格式（如 `{key: value}` 会展开为多行）。