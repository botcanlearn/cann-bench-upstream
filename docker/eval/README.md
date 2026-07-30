# `cann-bench` docker/eval -- 评测镜像:`docker run` 即评测

harness 冻结在镜像里(`src/kernel_eval` + `tasks/` + `cann_bench_utils` 全部烘入
`/opt/cann-bench`),外部只挂两样东西:

| 容器路径 | 内容 | 挂法 |
|---|---|---|
| `/submission` | AI 生成的算子源码目录 | 只读即可(entrypoint 会先复制) |
| `/reports` | 评测报告 + `prof_data/` + `build/`(编译日志、wheel) | 读写 |

**镜像 tag 就是 benchmark 版本**:`cann-bench-eval:<VERSION>-<NPU_ARCH>-ops<OPS_MODE>`,
例如 `cann-bench-eval:1.0.0-ascend910b-opsnone`。

## Build

底座是 [`docker/base`](../base/) 的 `cann-toolkit-base`,先有它:

```bash
cd docker/base && docker build -t cann-toolkit-base:9.0.1-py3.13 .
```

然后(**build context 必须是仓库根**,`build.sh` 已经处理好):

```bash
bash docker/eval/build.sh                       # 默认: OPS_MODE=none, ascend910b
OPS_MODE=refonly bash docker/eval/build.sh      # 见下"ops 模式"
NPU_ARCH=ascend910_93 bash docker/eval/build.sh # A3
MIRROR=cn bash docker/eval/build.sh             # 受限网络: 一把切到在区镜像源
TRITON_ASCEND_VERSION=3.2.1 bash docker/eval/build.sh
```

| build-arg | 默认 | 说明 |
|---|---|---|
| `BASE_IMAGE` | `cann-toolkit-base:9.0.1-py3.13` | 底座 |
| `OPS_MODE` | `none` | `none` / `refonly` / `full`,见下 |
| `NPU_ARCH` | `ascend910b` | `cann_bench_utils` 的 kernel 是 SoC 相关的,故本镜像 per-SoC |
| `TRITON_ASCEND_VERSION` | 空 | 非空则装 Triton-Ascend(体积大),语义同 `docker/dev` |
| `PYPI_MIRROR` / `TORCH_MIRROR` / `UV_PYTHON_INSTALL_MIRROR` | 空(官方源) | 同 `docker/base`;`MIRROR=cn` 是这三个的快捷方式 |

python 依赖由 `docker/eval/{pyproject.toml,uv.lock}` 锁定,是 `docker/base` 依赖集的**严格超集**
(`uv sync` 会把环境对齐到 lock —— 非超集会把底座已装的 torch 卸掉)。改任一边都要同步另一边。

## Run

```bash
bash docker/eval/run.sh self-test                                # 镜像自检
bash docker/eval/run.sh /path/to/ai_ops                          # 全量评测一个提交
bash docker/eval/run.sh /path/to/ai_ops --operator Exp --no-perf # 单算子, 仅精度
bash docker/eval/run.sh -- --task-dir tasks/level1 --no-perf     # 无提交(golden 自评)
bash docker/eval/run.sh shell                                    # 交互 shell, NPU 已绑入

REPORTS=/data/out ASCEND_RT_VISIBLE_DEVICES=0 bash docker/eval/run.sh /path/to/ai_ops
```

`run.sh` 只是把 NPU device flags 和两个挂载拼好;裸 `docker run` 完全等价:

```bash
docker run --rm --privileged --ipc=host \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -e LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64 \
  -v /path/to/ai_ops:/submission:ro -v "$PWD/reports:/reports" \
  cann-bench-eval:1.0.0-ascend910b-opsnone --operator Exp
```

### entrypoint 契约

```
docker run <image> [源码目录] [选项]
```

- `[源码目录]` 是**容器内**路径;省略则用 `/submission`。第一个参数不以 `-` 开头才算源码目录
  —— 这和 `scripts/run_evaluation.sh` 自己的约定一致(否则 `--operator Exp` 的 `Exp` 会被误判)。
- `[选项]` 原样透传给 `run_evaluation.sh`(`--operator` / `--case-id` / `--task-dir` /
  `--no-perf` / `--device-id` / `--warmup` / `--repeat` / `--eval-seed` / ...),
  完整表:`docker run <image> --help`。
- `--reports-dir` 由 entrypoint 强制注入 `/reports`。不注入的话 `run_evaluation.sh` 会写
  `${PROJECT_ROOT}/reports`,也就是镜像里那份冻结的 harness。
- **源码目录默认先复制到容器内 `/work/src` 再编译**。因为 `PackageManager` 是就地构建的:
  它 `rmtree` `<src>/{build,dist,*.egg-info}`、写 `<src>/_compile.log`,`build.sh` 的 cwd 就是
  源码目录。复制之后 `/submission` 可以 `:ro` 挂载,host 上那棵树也不会被反复跑脏。
  `--in-place` 关掉复制。收尾时 `_compile.log` 和 `dist/` 会被收到 `/reports/build/`(失败也收,
  编译错误的 bisheng/g++ 诊断只在那里面)。
- 显式给了 `--source-dir` 就原样尊重,不复制。
- 逃生口(作为第一个参数):`bash` / `sh` / `python3` / `python` / `uv` / `pip` 直接执行;
  `--self-test` 跑自检;`--help` 打用法。

### 开发时覆盖冻结的 harness

改 `kernel_eval` 时不必重 build:

```bash
docker run ... -v "$PWD:/opt/cann-bench" cann-bench-eval:... /submission --operator Exp
```

## ops 模式(反作弊形态)

提交"作弊"的方式是**调用内置算子**(`aclnn<Op>` / `torch_npu` op),它们下发到 4.2G 的
`opp/built-in/op_impl/ai_core/tbe/kernel` 二进制树。旁边 151M 的 `tbe/impl` 是 AscendC **源码**,
那不是作弊,是合法参考。

| `OPS_MODE` | 做什么 | 后果 |
|---|---|---|
| `none`(默认) | 不装 ops(底座本来就是 0-ops) | 内置算子根本不存在,无从蹭起;镜像最小。`opp/` 目录仍在(toolkit 自带),`.run` 形态的自定义算子提交照常装进 `opp/vendors` |
| `refonly` | 装 ops,**同层**删掉 `tbe/kernel` 二进制 | 保留 `tbe/impl` AscendC 源码作参考,内置算子下发失败;比 `none` 多出 opp 的全套机制 |
| `full` | 装 ops 不删 | 可被蹭内置算子;用于重采 aclnn baseline |

自检的 `[6]` 项会直说当前镜像里内置算子能不能下发。

**默认 `none` 够跑全量三阶段(编译/精度/性能)** —— 910B2 上实测:`direct_launch_example` 的 Sqrt
4/4 精度通过,profiler 正常产出 `prof_data/` 与 device kernel 耗时(`sqrt_kernel` 6.9us),综合得分
73.00。`LazyInitAclops` 在 0-ops 下确实会失败(自检 `[6]` 就是它),但性能采集不经过这条路 ——
升频/清 cache 由镜像里烘好的 `cann_bench_utils` 直调 kernel 提供。所以 `refonly` 只在提交本身
需要 opp 全套机制时才用得上。

## 已知取舍

- 底座的 `torch_npu==2.10.0.post2` 与仓库 `pyproject.toml` 钉的 `2.10.0` 不是同一串号(同属 2.10
  系)。沿用底座的 post2,避免为了对齐版本号把整个 torch 栈重装一遍。
- `tasks/` 烘进镜像,所以镜像 tag 即 benchmark 版本;开发期用上面的 `-v` 覆盖回工作树。
- `cann_bench_utils` 在 build 期就编好装好(只要 bisheng,不需要 NPU),容器起来直接开跑,
  `ensure_cann_bench_utils()` 短路返回。它含 SoC 相关 kernel,所以本镜像 per-SoC。
