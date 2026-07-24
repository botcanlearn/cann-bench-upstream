# `cann-bench` — 多平台 CI 执行镜像

CANN-BENCH 参考执行镜像。torch 2.10.0 + torch\_npu 2.10.0 + 相关科学计算栈。
通过 `--build-arg` 参数化 CANN 版本和硬件型号，同一 Dockerfile 适配不同设备。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CANN_VERSION` | `9.0.0` | CANN toolkit 版本 |
| `DEVICE` | `910b` | 硬件型号（910b / 910c / 950 等，对应 AscendHub 镜像 tag） |
| `TRITON_ASCEND_VERSION` | 空 | 可选 Triton-Ascend 版本；当前验证版本为 `3.2.1` |

默认 tag `cann-bench:cann9.0.0-910b-latest`。

- `Dockerfile` — 镜像定义（ARG 参数化）
- `entrypoint.sh` — 容器入口 (source CANN env, 转交 CMD)
- `run.sh` — host 端 launcher (smoke / shell / dev 三种模式)
- `test_env.py` — smoke 验证脚本 (版本 / torch\_npu device / npu-smi / CANN)

## 1. Build image

在 NPU HOST 上 build image:

### 910B（默认）

```bash
cd /path/to/repo/docker/
docker build --network=host -t cann-bench:cann9.0.0-910b-latest .
```

### 950PR

```bash
docker build --network=host \
    --build-arg CANN_VERSION=9.0.0 --build-arg DEVICE=950 \
    -t cann-bench:cann9.0.0-950-latest .
```

### Triton-Ascend（910B / 950PR）

```bash
docker build --network=host \
    --build-arg CANN_VERSION=9.0.0 \
    --build-arg DEVICE=950 \
    --build-arg TRITON_ASCEND_VERSION=3.2.1 \
    -t cann-bench:cann9.0.0-950-triton3.2.1 .
```

设置该参数后，镜像 smoke 会实际 JIT 编译并运行 Triton vector add，而不只是检查 import。

也可配置代理:
```bash
docker build --network=host \
    --build-arg HTTP_PROXY --build-arg HTTPS_PROXY \
    -t cann-bench:cann9.0.0-910b-latest .
```

也可配置 pypi 镜像源: `--build-arg PYPI_INDEX_URL`。

## 2. Smoke

验证 python / torch / torch\_npu / npu-smi / CANN 全 OK:

```bash
bash run.sh smoke
# 或指定 950PR 镜像:
IMAGE=cann-bench:cann9.0.0-950-latest bash run.sh smoke
# Triton-Ascend image:
IMAGE=cann-bench:cann9.0.0-950-triton3.2.1 bash run.sh smoke
```

期望 `ALL CHECKS PASSED`。

## 3. 启动临时容器

退出即删:

```bash
bash run.sh shell
```

## 4. 启动常驻容器

后台 `sleep infinity`, 多次 `docker exec` 进入; `docker/workspace/` 绑到容器内 `/workspace`:

```bash
bash run.sh dev                          # 起 'cann-bench'
docker exec -it cann-bench bash
docker rm -f cann-bench                  # 收尾
```

Override: `CONTAINER=<name> WORKSPACE=<host-path> bash run.sh dev`。

## Env

| 变量        | 默认                              |
|-------------|-----------------------------------|
| `IMAGE`     | `cann-bench:cann9.0.0-910b-latest` |
| `CONTAINER` | `cann-bench` (仅 dev)             |
| `WORKSPACE` | `$(pwd)/workspace` (仅 dev)       |
