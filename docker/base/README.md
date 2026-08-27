# `cann-bench` docker/base -- 最小 toolkit-only 直调基础镜像

与 `docker/dev`(AscendHub 完整 CANN,含 ops/nnal 的调试/CI 镜像)**互补**,并且是 `docker/eval`
(评测镜像)的**底座**。本镜像只装
CANN **toolkit**(bisheng + AscendC 头 + acl runtime)+ torch(cpu)/torch_npu 胶水,**不装 ops**
(`libopapi` + kernel blob)/nnal,面向 **直调(direct-launch)风格提交** -- 提交自带 bisheng 编译的
AscendC/CCE kernel,经 `KNAME<<<grid, nullptr, stream>>>` 直接下发,不走 aclnn,故 ops 是死重。

| | `docker/dev`(AscendHub 全量) | `docker/base`(本镜像) |
|------|------|------|
| base | AscendHub 完整 CANN(`cann:<ver>-<device>-...`,per-device) | `debian:12-slim`,从 `.run` 自装 toolkit |
| ops / nnal | 有 | **无(0 ops)** |
| chip | per-`DEVICE` tag | **chip-agnostic**(chip 只进 mounted driver + bisheng `--soc`) |
| CPU 架构 | per-tag | `ARCH` build-arg(aarch64 / x86_64),原生构建;写入 `ENV CANN_ARCH` 供下游继承 |
| py / env | ubuntu22.04 + py3.12 | uv 管理的 py3.13 standalone(`uv.lock` 锁定) |
| 适用 | 全量评测(含 aclnn baseline + perf 开箱) | 直调提交:精度独立可跑;perf 见下 |

## 为什么(直调 / 反作弊)

直调 kernel 自带实现,不蹭内置优化算子 -- 这正是反作弊(anti-cheat)想要的形态。本镜像根本不含
内置 kernel 树,提交无从蹭起;相比在完整镜像上运行时搬走 4.2G kernel 树的做法(见
`scripts/anti_cheat/`),这里天然如此,且镜像更小、chip-agnostic、自包含(不依赖体积大且受限的
AscendHub per-device 镜像)。

## Build

```bash
cd docker/base/
docker build --build-arg ARCH=$(uname -m) -t cann-toolkit-base:9.1.0-py3.13 .
```

镜像架构由 **`ARCH` build-arg** 决定(`aarch64` / `x86_64`,默认 `aarch64`),**必须在目标架构的
机器上原生构建** —— 没有交叉编译、没有 `--platform`。两种架构都已实测跑通(aarch64/910B2 与
x86_64/950PR:装 toolkit、同一份 lock `uv sync --frozen`、stub libhccl、编 `cann_bench_utils`、
认卡)。构建结果会把架构写进 `ENV CANN_ARCH`,`docker/eval` **继承**它来挑对应架构的 ops 包 ——
所以下游不该、也不需要再声明自己的 `ARCH`。

python 依赖由
`pyproject.toml` + `uv.lock` 锁定(hash 校验),`uv sync --frozen` 装入 `/opt/venv`。

### 镜像源(每个都默认走官方/全球源;受限网络用 `--build-arg` 换在区镜像)

| build-arg | 默认(官方) | 换镜像示例(CN) |
|------|------|------|
| `BASE_OS` | `debian:12-slim`(docker.io) | `docker.m.daocloud.io/library/debian:12-slim` |
| `UV_IMAGE` | `ghcr.io/astral-sh/uv:0.11.29` | `ghcr.m.daocloud.io/astral-sh/uv:0.11.29` |
| `APT_MIRROR` | (空 = `deb.debian.org`) | `mirrors.huaweicloud.com` |
| `UV_PYTHON_INSTALL_MIRROR` | (空 = github releases) | `https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone` |
| `PYPI_MIRROR` | (空 = `files.pythonhosted.org`) | `https://mirrors.huaweicloud.com/repository/pypi` |
| `TORCH_MIRROR` | (空 = `download.pytorch.org`) | `https://mirror.nju.edu.cn/pytorch/whl/cpu` |
| `CANN_VERSION` |  `9.1.0` | toolkit `.run` 版本(从 OBS 拉取);落到 `ENV CANN_VERSION` 供 `docker/eval` 继承 |
| `CANN_TOOLKIT_URL` | 空(由 `CANN_VERSION` + `ARCH` 推导) | 只在该版本不按常规命名/不在常规 bucket 时设,见下 |
| `ARCH` | `aarch64` | `aarch64` / `x86_64`;决定 toolkit `.run`、`<arch>-linux/` 路径,并落到 `ENV CANN_ARCH` |

### 换 CANN 版本

正常只需 `--build-arg CANN_VERSION=<版本>` —— toolkit 和 ops 两个包在
`ascend-repo.obs.cn-east-2` 上的命名跨版本一致,`docker/eval` 会继承 `ENV CANN_VERSION`
去推导同版本的 ops:

```bash
# base
docker build --build-arg CANN_VERSION=<版本> --build-arg ARCH=$(uname -m) \
             -t cann-toolkit-base:<版本>-py3.13 .
# eval (build.sh 用 CANN_VERSION 只是为了拼 BASE_IMAGE 的 tag)
CANN_VERSION=<版本> NPU_ARCH=ascend950 bash ../eval/build.sh
```

**换版本时 `torch_npu` 要按[官方配套表](https://gitcode.com/Ascend/pytorch/blob/master/COMPATIBILITY.md)
重新选**:把 `docker/base/pyproject.toml` 与 `docker/eval/pyproject.toml` 的 `torch-npu` 改成
新 CANN 版本在表里对应的串号,再重新 `uv lock`。表里没有对应行的 CANN 版本不要用 —— 那样钉出来
的组合官方没有验证过,理由见下一节。

`CANN_TOOLKIT_URL` 是给例外准备的:部分版本发在 `ascend-cann-open.obs.cn-north-4` 上,
或以合并包 `Ascend-cann_<版本>_linux-<arch>.run` 的形式发布(而非 `Ascend-cann-toolkit_...`)。

### 为什么是 9.1.0 + torch_npu 2.10.0.post4

[官方配套表](https://gitcode.com/Ascend/pytorch/blob/master/COMPATIBILITY.md)里 torch_npu
2.10 这条线**只配 CANN 9.1.0**(`2.10.0.post4` ↔ torch `2.10.0` ↔ CANN `9.1.0`),表里没有
9.0.1 这一行,`2.10.0.post2` 也不在表内。所以本镜像此前的 `9.0.1 + 2.10.0.post2` 是表外组合;
现在这一档是第一次落在表内。Python 3.13 在该行的支持范围内。

9.1.0 的 toolkit/ops × aarch64/x86_64 四个包均已核实存在。实测(q7 / 910B2 / aarch64):
base + eval(`OPS_MODE=refonly`,顺带验了 9.1.0 的 ops 包)全部建成,`--self-test` 全绿、
内置算子仍被挡住,一次真实三阶段评测 4/4 通过、得分 74.10(9.0.1 同一提交同一算子是 73.00)。
安装布局也没变(`cann-9.1.0/` + `latest` 符号链接 + `ascend-toolkit/set_env.sh`)。

`PYPI_MIRROR`/`TORCH_MIRROR` 就地改写 `uv.lock` 里的 canonical wheel URL(**同 hash**,`--frozen` 仍校验),
换源不破坏可复现性。CN 全量示例:

```bash
docker build -t cann-toolkit-base:9.1.0-py3.13 \
  --build-arg BASE_OS=docker.m.daocloud.io/library/debian:12-slim \
  --build-arg APT_MIRROR=mirrors.huaweicloud.com \
  --build-arg UV_IMAGE=ghcr.m.daocloud.io/astral-sh/uv:0.11.29 \
  --build-arg UV_PYTHON_INSTALL_MIRROR=https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone \
  --build-arg PYPI_MIRROR=https://mirrors.huaweicloud.com/repository/pypi \
  --build-arg TORCH_MIRROR=https://mirror.nju.edu.cn/pytorch/whl/cpu .
```

## Run(NPU host)

`run.sh` 负责 device 挂载 + 把 host driver runtime libs 放上 `LD_LIBRARY_PATH`(dev 镜像从
AscendHub base 继承,本镜像需显式设):

```bash
bash run.sh smoke     # torch_npu device_count / name
bash run.sh shell     # 交互 shell,NPU 已绑入
bash run.sh dev       # 后台 sleep infinity,供 docker exec 调试
```

## 评测直调提交

挂载本仓库(`src/` + `tasks/`)+ 提交源码进容器,跑 `scripts/run_evaluation.sh`:

- **精度**(`--no-perf`):**0 ops 即可,独立可跑**。golden 走 CPU,提交 kernel 走 NPU,CPU 对比。
- **性能**:另需两项(均 lean-side,非 ops)——
  - 框架 warmup 算子 `cann_bench_utils`:无 ops 镜像上内置 `torch.matmul` / `torch.max`
    (升频 / 清 cache)因 `LazyInitAclops` 不可用,需自定义**直调** warmup 顶替(参见 PR #207)。
  - `libsqlite3`:**已烘入本镜像** -- msprof 导出 `kernel_details.csv`(perf stage 经
    `torch_npu.profiler` 触发)`import sqlite3` 需要它,`debian-slim` 默认不带。

## 不适用

- **aclnn baseline** 或任何需内置算子的评测:内置 torch_npu 算子在本镜像上 `LazyInitAclops` 失败 ->
  走 `docker/dev` 镜像(需 ops),或 `docker/eval --build-arg OPS_MODE=full`。本镜像专供直调 kernel。
