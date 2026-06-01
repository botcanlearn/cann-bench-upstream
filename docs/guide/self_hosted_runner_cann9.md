# Self-Hosted GitHub Runner（NPU / CANN 9.0.0）部署指南

> 目标：在自定义 Ascend NPU 服务器上以容器化方式部署 GitHub Actions self-hosted runner，跑 `inner/tests/baseline/` 这条 NPU-bound 测试链路（msprof bench / drift gate）。

- **版本**: V0.1（Draft）
- **日期**: 2026-05-15
- **适用 runner workload**: `pytest inner/tests/baseline/` + `python inner/tests/apply_baselines.py --check`
- **不适用**: CPU-only spec 测试（已在 ubuntu-latest 跑，见 `.github/workflows/validate-tasks.yml`）

需用户根据实际环境填写的 placeholder 用 `<...>` 标注。

---

## 1. 架构

```
┌──────────────────────────────────────────────────────────────────────┐
│  NPU 服务器（物理机）                                                │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  OS: <openEuler 22.03 LTS / Ubuntu 22.04>                       │ │
│  │  NPU 驱动: Ascend HDK <version>                                  │ │
│  │  Container runtime: Docker 24.x + ascend-docker-runtime         │ │
│  │  ┌───────────────────────────────────────────────────────────┐  │ │
│  │  │ Container: cann-bench-runner                              │  │ │
│  │  │  Base image: ascendai/cann:9.0.0-<arch>-<os>-py3.10        │  │ │
│  │  │  ├─ /opt/Ascend/ (CANN toolkit + msprof)                  │  │ │
│  │  │  ├─ /opt/runner/ (GitHub Actions runner binary)           │  │ │
│  │  │  ├─ /work/cann-bench-dev/ (mounted repo / workdir)        │  │ │
│  │  │  └─ device passthrough: /dev/davinci0 (+ HCCL devices)    │  │ │
│  │  └───────────────────────────────────────────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  github.com/<org>/cann-bench-dev  ◄────── HTTPS poll (run job) ───── │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. 前置要求

### 2.1 硬件

| 项 | 最低 | 推荐 |
|---|---|---|
| NPU | 1× Ascend 910B / 910A / 310P / 950 | 1× Ascend 910B（与 cann-bench baseline 历史数据一致的 a3 平台） |
| CPU | 16 core | 32+ core |
| RAM | 64 GB | 128 GB |
| 磁盘 | 200 GB SSD | 500 GB NVMe（msprof profile 数据 / runner 缓存） |
| 网络 | 出方向访问 `github.com`、`api.github.com`、`pypi.org` | 同左，建议有 Huawei mirror 加速 |

### 2.2 软件

| 组件 | 版本 | 验证命令 |
|---|---|---|
| OS kernel | Linux ≥ 5.10 | `uname -r` |
| Ascend NPU 驱动 | 与 CANN 9.0.0 兼容（详见华为 Ascend 兼容矩阵） | `npu-smi info` 能看到所有卡 |
| CANN 9.0.0 toolkit | 8.0 / 7.0 不可用 | `cat /opt/Ascend/ascend-toolkit/latest/version.info` |
| Docker | ≥ 24.0 | `docker --version` |
| ascend-docker-runtime | 与驱动配套 | `docker info \| grep -A2 Runtimes` 应能看到 `ascend` |
| msprof | 与 CANN 9.0.0 配套（容器内） | `which msprof && msprof --version` |

> **关于 CANN 9.0.0**：本仓库 baseline 历史数据是在 CANN <历史版本，需用户填写> 上测出。升级到 9.0.0 后**所有 baseline_perf_us 数字可能漂移**（kernel dispatch 路径变化），第一次跑 `apply_baselines --check` 时务必带 `--tolerance` 调宽，或直接 `--apply --force` rebaseline。

### 2.3 凭据 / 权限

- **GitHub repo admin 权限**：用于注册 runner、生成 token
- **Runner registration token**：在 `https://github.com/<org>/cann-bench-dev/settings/actions/runners` → `New self-hosted runner` 获取（**注意：token 24 小时过期，仅一次性 register 用，注册成功后会自动持久化为 .runner credential**）
- **可选：自定义 secrets**（如 PyPI mirror token、msprof license 等）以 GitHub Actions secrets 注入

---

## 3. 主机准备

### 3.1 安装 NPU 驱动 + CANN runtime（宿主机层，仅 driver）

按华为 Ascend HDK 文档安装与 CANN 9.0.0 配套的 NPU 驱动包（`Ascend-hdk-<chipset>-npu-driver_*.run`）+ firmware。**toolkit 不需要装宿主机**——toolkit 放容器里。

验证：

```bash
npu-smi info       # 应显示所有可见 NPU 卡，状态 OK
ls /dev/davinci*   # 应见 /dev/davinci0..N + /dev/davinci_manager + /dev/devmm_svm + /dev/hisi_hdc
```

### 3.2 安装 ascend-docker-runtime

按华为 Ascend Docker Runtime 文档安装（包名 `Ascend-docker-runtime_*.run` 或 deb/rpm）。安装后修改 `/etc/docker/daemon.json`：

```json
{
  "default-runtime": "ascend",
  "runtimes": {
    "ascend": {
      "path": "/usr/local/Ascend/Ascend-Docker-Runtime/ascend-docker-runtime",
      "runtimeArgs": []
    }
  }
}
```

重启 docker：`sudo systemctl restart docker`。

验证：

```bash
docker info | grep -A2 "Runtimes:"
# Runtimes: ascend runc
# Default Runtime: ascend
```

### 3.3 拉 CANN 9.0.0 基础镜像

```bash
# 镜像 tag 以华为官方发布为准，下面是 placeholder
docker pull ascendai/cann:9.0.0-<arch>-<os>-py3.10
# 例如 ascendai/cann:9.0.0-910b-openeuler22.03-py3.10
```

如果无法直接拉取，参考华为 ModelZoo / 内部 mirror 文档自构建：基础 `ubuntu:22.04` + CANN 9.0.0 toolkit `.run` 安装包 + Python 3.10。

---

## 4. 构建 runner 镜像

### 4.1 Dockerfile

放仓库 `infra/runner/Dockerfile`（**新增目录，不在本轮 inner/tests/ 整合 scope 内，需单独提 PR**）：

```dockerfile
# syntax=docker/dockerfile:1.6
ARG CANN_IMAGE=ascendai/cann:9.0.0-<arch>-<os>-py3.10
FROM ${CANN_IMAGE}

ARG RUNNER_VERSION=2.319.0
ARG TARGETARCH=x86_64

# --- 系统依赖 -------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl jq git sudo tini libicu70 \
    && rm -rf /var/lib/apt/lists/*

# --- GitHub Actions runner ------------------------------------------------
RUN useradd -m -s /bin/bash runner && \
    mkdir -p /opt/runner && chown runner:runner /opt/runner

USER runner
WORKDIR /opt/runner
RUN curl -fsSL -o runner.tgz \
        "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-${TARGETARCH}-${RUNNER_VERSION}.tar.gz" \
    && tar xzf runner.tgz \
    && rm runner.tgz

# --- repo Python 依赖 (CPU 部分；torch_npu 由 CANN 镜像提供) --------------
COPY --chown=runner:runner requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --user -r /tmp/requirements.txt \
    && pip3 install --no-cache-dir --user ruamel.yaml pytest pytest-timeout

# --- entrypoint -----------------------------------------------------------
COPY --chown=runner:runner infra/runner/entrypoint.sh /opt/runner/entrypoint.sh
RUN chmod +x /opt/runner/entrypoint.sh

ENV PATH="/home/runner/.local/bin:/opt/Ascend/ascend-toolkit/latest/bin:/opt/Ascend/ascend-toolkit/latest/tools/profiler/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/Ascend/ascend-toolkit/latest/lib64:/opt/Ascend/driver/lib64:${LD_LIBRARY_PATH}"
ENV ASCEND_TOOLKIT_HOME="/opt/Ascend/ascend-toolkit/latest"
ENV PYTHONPATH="/opt/Ascend/ascend-toolkit/latest/python/site-packages:${PYTHONPATH}"

ENTRYPOINT ["/usr/bin/tini", "--", "/opt/runner/entrypoint.sh"]
```

### 4.2 entrypoint.sh

放仓库 `infra/runner/entrypoint.sh`：

```bash
#!/bin/bash
set -euo pipefail

: "${GITHUB_REPO_URL:?must set GITHUB_REPO_URL, e.g. https://github.com/<org>/cann-bench-dev}"
: "${RUNNER_REG_TOKEN:?must set RUNNER_REG_TOKEN (24h registration token)}"

RUNNER_NAME="${RUNNER_NAME:-$(hostname)-$(date +%s)}"
RUNNER_LABELS="${RUNNER_LABELS:-self-hosted,npu,cann9,ascend910b}"
RUNNER_WORKDIR="${RUNNER_WORKDIR:-/work}"

cd /opt/runner

# 注册（仅首次；.runner 文件存在则跳过）
if [[ ! -f .runner ]]; then
    ./config.sh \
        --unattended \
        --url "$GITHUB_REPO_URL" \
        --token "$RUNNER_REG_TOKEN" \
        --name "$RUNNER_NAME" \
        --labels "$RUNNER_LABELS" \
        --work "$RUNNER_WORKDIR" \
        --replace
fi

# 退出/SIGTERM 时自动注销，避免 GitHub UI 残留 offline runner
cleanup() {
    ./config.sh remove --unattended --token "$RUNNER_REG_TOKEN" || true
}
trap cleanup EXIT INT TERM

exec ./run.sh
```

### 4.3 构建

```bash
cd /path/to/cann-bench-dev
docker build \
    --build-arg CANN_IMAGE=ascendai/cann:9.0.0-<arch>-<os>-py3.10 \
    --build-arg RUNNER_VERSION=2.319.0 \
    -t cann-bench-runner:cann9.0.0 \
    -f infra/runner/Dockerfile .
```

---

## 5. Runner 注册 + 启动

### 5.1 获取 registration token

GitHub UI: `https://github.com/<org>/cann-bench-dev/settings/actions/runners` → `New self-hosted runner` → 复制 `--token` 后的值。**24h 过期，注册完毕即可。**

或用 API（需要 `repo` scope PAT）：

```bash
TOKEN=$(curl -sfL -X POST \
    -H "Authorization: token $GITHUB_PAT" \
    -H "Accept: application/vnd.github+json" \
    https://api.github.com/repos/<org>/cann-bench-dev/actions/runners/registration-token \
    | jq -r .token)
```

### 5.2 启动 container

```bash
docker run -d \
    --name cann-bench-runner-0 \
    --restart unless-stopped \
    --runtime ascend \
    --device /dev/davinci0 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v /var/log/cann-bench-runner-0:/work/_diag \
    -e GITHUB_REPO_URL="https://github.com/<org>/cann-bench-dev" \
    -e RUNNER_REG_TOKEN="$TOKEN" \
    -e RUNNER_NAME="cann-bench-npu0" \
    -e RUNNER_LABELS="self-hosted,npu,cann9,ascend910b,device0" \
    --shm-size 16g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    cann-bench-runner:cann9.0.0
```

**关键点**：

- `--runtime ascend` 让 ascend-docker-runtime 处理 NPU 设备透传（自动注入更多 device 节点）
- `--device /dev/davinci0` 指定哪张卡。多卡场景见 §7.2
- `-v` 三条挂载驱动 / dcmi / npu-smi：容器内才能调用 `npu-smi`
- `--shm-size 16g`：msprof / pytorch dataloader 需要大 shm
- `--ulimit memlock=-1`：避免 ACL memory pin 受限
- `RUNNER_LABELS` 里加 `device0` 是为了多卡 runner 之间能用 `runs-on` 精确路由

### 5.3 验证

```bash
# 进容器看启动日志
docker logs -f cann-bench-runner-0
# 应看到:
#   √ Connected to GitHub
#   2026-... Listening for Jobs

# GitHub UI 上 Settings → Actions → Runners 应显示 "cann-bench-npu0  Idle"

# 容器内验证 NPU 可见
docker exec cann-bench-runner-0 npu-smi info
docker exec cann-bench-runner-0 python3 -c "import torch, torch_npu; print(torch_npu.npu.device_count())"
```

---

## 6. Workflow 适配

在 `.github/workflows/` 下新建 `baseline-bench.yml`（**不在 PR auto-trigger**，仅手动 / 定时跑）：

```yaml
name: baseline bench (NPU)

on:
  workflow_dispatch:
    inputs:
      mode:
        description: "check (drift gate) or apply (rebaseline)"
        required: true
        default: "check"
        type: choice
        options: [check, apply]
      tolerance:
        description: "drift tolerance for --check (fraction)"
        required: false
        default: "0.20"
  schedule:
    # 每周一 02:00 UTC 自动跑 --check
    - cron: "0 2 * * 1"

jobs:
  bench:
    runs-on: [self-hosted, npu, cann9]      # 通过 labels 路由到我们这台 runner
    timeout-minutes: 480                     # ≈ 8h，覆盖 6 op × 20 case 全跑 + 余量
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Verify NPU visible
        run: |
          npu-smi info
          python3 -c "import torch_npu; print('npu count =', torch_npu.npu.device_count())"

      - name: Run baseline pytest
        env:
          CANN_BENCH_DEVICE: "0"
        run: |
          pytest inner/tests/baseline/ -v --timeout=300 \
              --junitxml=inner/tests/baseline/results/junit.xml

      - name: Drift check / apply
        run: |
          MODE="${{ github.event.inputs.mode || 'check' }}"
          TOL="${{ github.event.inputs.tolerance || '0.20' }}"
          set -euo pipefail
          if [ "$MODE" = "apply" ]; then
            # manual dispatch + mode=apply 是显式 rebaseline (覆盖已有 baseline);
            # 默认 --apply 只填 null 项, 不带 --force 看似 succeed 实则 no-op.
            python3 inner/tests/apply_baselines.py \
                --input inner/tests/baseline/results/baseline_perf_*.json \
                --apply --force
          else
            python3 inner/tests/apply_baselines.py \
                --input inner/tests/baseline/results/baseline_perf_*.json \
                --check --tolerance "$TOL"
          fi

      - name: Upload results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: baseline-results-${{ github.run_id }}
          path: |
            inner/tests/baseline/results/
```

**注意：不要把 `inner/tests/baseline/` 加进 `pull_request` 触发**。原因见 §8.1（fork PR 安全）。

---

## 7. 运维

### 7.1 日志 / 健康检查

| 检查项 | 命令 |
|---|---|
| 容器存活 | `docker ps --filter name=cann-bench-runner-` |
| Runner 心跳 | `docker logs --tail 50 cann-bench-runner-0 \| grep Listening` |
| NPU 占用 | `npu-smi info -t usages` |
| 磁盘 | `df -h /var/lib/docker /work` |

建议加 systemd timer + Prometheus node-exporter，详见 [华为 Ascend 监控文档]（用户根据内部规范补）。

### 7.2 多卡多 runner（每张卡一个 container instance）

```bash
for i in 0 1 2 3; do
    docker run -d \
        --name cann-bench-runner-$i \
        --restart unless-stopped \
        --runtime ascend \
        --device /dev/davinci$i \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
        ...
        -e RUNNER_NAME="cann-bench-npu$i" \
        -e RUNNER_LABELS="self-hosted,npu,cann9,ascend910b,device$i" \
        cann-bench-runner:cann9.0.0
done
```

workflow 用 `runs-on: [self-hosted, npu, device0]` 精确路由到某张卡。

### 7.3 升级

升 CANN 版本：

1. 拉新基础镜像
2. 改 Dockerfile `CANN_IMAGE` arg
3. `docker build` 新 tag
4. 滚动重启 container：`docker stop cann-bench-runner-0 && docker rm cann-bench-runner-0 && docker run ... cann-bench-runner:cann9.x.0`
5. **强制 rebaseline**：因为 ACLNN kernel dispatch 可能变化，`python inner/tests/apply_baselines.py --apply --force`，并 commit 新 cases.yaml

升 runner 二进制：改 Dockerfile `RUNNER_VERSION`，重新 build。

### 7.4 Registration token 轮换

`.runner` credential 文件在 container `/opt/runner/.runner` 里（持久于 container layer，重启不丢）。重建 container 时 entrypoint 会自动重新 register（需要重新生成 token）。生产建议把 `/opt/runner/.runner` + `/opt/runner/.credentials*` 挂出来：

```bash
docker run ... \
    -v /var/lib/cann-bench-runner-0/runner-creds:/opt/runner/.runner-creds \
    ...
# entrypoint 改造为优先用挂载的 credential, 若不存在再 register
```

---

## 8. 安全

### 8.1 阻断 fork PR 触发 self-hosted runner（关键）

self-hosted runner 跑在自家 NPU 服务器上，**任何能触发它的代码都有 RCE 风险**。GitHub 默认 `pull_request` 不让 fork PR 拿到 secrets，但 **workflow 代码本身**可以被 PR 改——攻击者 PR 改 workflow 为 `pip install evil-package`，runner 就会跑。

防御：

1. Repo Settings → Actions → General → "Fork pull request workflows from outside collaborators" 设为 **Require approval for all outside collaborators**
2. baseline-bench workflow **只用 `workflow_dispatch` + `schedule`**，不挂 `pull_request`
3. 若需要 PR 触发，用 `pull_request_target` + 显式 checkout `github.event.pull_request.base.sha`，**不**checkout PR HEAD

### 8.2 凭据隔离

- `RUNNER_REG_TOKEN` 用一次性 env var 注入，**不**打镜像里
- repo secrets（PAT 等）只对 `workflow_dispatch` 触发的 job 暴露
- runner container 不挂 `~/.ssh` / `~/.aws` / `/root`

### 8.3 NPU 资源隔离

多 runner 共享主机时：

- 每个 runner container 绑定 1 张卡（`--device /dev/davinci$i`），不要共享
- container 内 `CANN_BENCH_DEVICE=0`（永远指容器看到的 device 0，即被绑定的物理卡）
- `--cpus`、`--memory` 限定避免单 job OOM 影响其他 container

---

## 9. 故障排查

| 现象 | 可能原因 | 修复 |
|---|---|---|
| `docker run` 报 `runtime ascend not found` | ascend-docker-runtime 未装 / daemon 未重启 | §3.2 |
| 容器内 `npu-smi info` 报 `unable to initialize device` | driver/runtime 版本不匹配 | 对照华为兼容矩阵 |
| `torch_npu.npu.device_count()` 返回 0 | device 未透传 / `LD_LIBRARY_PATH` 缺驱动 lib | 检查 `--device` flags + `/usr/local/Ascend/driver` 挂载 |
| Runner 注册成功但 GitHub UI 显示 offline | container 退出 / 网络出方向被防火墙拦 | `docker logs`；`curl -v https://api.github.com` |
| `msprof: command not found` | PATH 未包含 toolkit profiler | 容器内 `source /opt/Ascend/ascend-toolkit/set_env.sh`；或在 Dockerfile ENV 里加（本 doc Dockerfile 已加） |
| `--apply` 后 cases.yaml license header 丢失 | ruamel.yaml 版本问题 | 升级 ruamel.yaml ≥ 0.18，或参考 `apply_baselines.py` 内 assert 报错 |
| msprof export 慢 / 卡死 | profile 输出目录磁盘满 / 旧 profile 没清 | 清 `inner/tests/baseline/results/`；扩磁盘 |

---

## 10. 待用户决策的 placeholder 汇总

实际部署前请确认/填写下面这些值：

| Placeholder | 例子 | 出处 |
|---|---|---|
| `<arch>` | `910b` / `aarch64` / `x86_64` | §3.3, §4.1, Dockerfile ARG |
| `<os>` | `openeuler22.03` / `ubuntu22.04` | §3.3 |
| Ascend HDK 版本 | 由 CANN 9.0.0 兼容矩阵决定 | §2.2 |
| `<历史版本>` | 当前 baseline 测出时的 CANN tag | §2.2 |
| `<org>` | GitHub 组织或用户 | §5.1, §6 |
| Runner labels 命名规范 | 推荐 `self-hosted,npu,cann9,ascend910b,device<N>` | §5.2 |
| `infra/runner/` 是否纳入本仓主目录 | 默认放仓库根 `infra/runner/` | §4.1 |

---

## 11. 与本仓 inner/tests/ 整合的关系

本部署文档对应的 workload 来自：

- `inner/tests/baseline/test_baseline_perf.py` — msprof 实测 → `results/baseline_perf_<soc>_<ts>.json`
- `inner/tests/baseline/test_baseline_prec.py` — ref ↔ measured 精度对照
- `inner/tests/apply_baselines.py --check / --apply` — drift gate / rebaseline 回填

详见 [`docs/superpowers/specs/2026-05-15-inner-tests-integration-design.md`](../superpowers/specs/2026-05-15-inner-tests-integration-design.md)（如果该路径未提交至主仓，则查 `xc/dev` 分支本地 worktree）。

CPU-only 的 `inner/tests/task/` 链路**继续在 ubuntu-latest 跑**，不需要本 self-hosted runner。
