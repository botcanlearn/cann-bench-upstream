# CI image for the cann-bench golden-candidate NPU ST (self-contained: CANN + torch/torch_npu + pytest).
# Build (context = repo root):  bash tests/st/build_st_docker_image.sh
# Run ST on an NPU host:         bash tests/st/run_st_docker.sh     (see tests/st/run_st_docker.sh)
#
# Build deps are docker, python, uv only: torch/torch_npu are prebuilt wheels (no
# compiler/cmake needed) and uv is installed from PyPI. The repo is bind-mounted at
# /workspace at run time (not COPYed), so the image is repo-content-agnostic — rebuild
# only when the toolchain/deps change, not when code changes.
FROM swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-beta.1-910b-ubuntu22.04-py3.12-devel

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# PyPI mirror: Huawei Cloud (matches the Ascend Hub base image's region; the default PyPI
# wheel hosts are slow here). Overridable with `--build-arg PYPI_INDEX_URL=...`. Set for
# both uv (the installs below) and pip (uv itself + any runtime `uv run --with ...`).
ARG PYPI_INDEX_URL=https://mirrors.huaweicloud.com/repository/pypi/simple
ENV UV_INDEX_URL=${PYPI_INDEX_URL}
RUN pip config set --global global.index-url ${PYPI_INDEX_URL} && \
    pip config set --global global.trusted-host mirrors.huaweicloud.com && \
    pip install --no-cache-dir uv

# torch_npu install needs these at build time to locate driver libs.
ENV ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
ENV LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:${ASCEND_HOME_PATH}/aarch64-linux/lib64

RUN uv pip install --system torch==2.9.0
RUN uv pip install --system numpy pyyaml decorator scipy attrs cloudpickle \
    ml-dtypes psutil tornado pandas expecttest protobuf
RUN uv pip install --system torch_npu==2.9.0
RUN uv pip install --system pytest pytest-timeout 'ruamel.yaml>=0.18'

WORKDIR /workspace
