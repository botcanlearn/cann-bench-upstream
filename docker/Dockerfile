# cann-bench — CI execution image.
#
# 多平台支持：通过 --build-arg 切换 CANN 版本和硬件型号。
#
# 示例：
#   docker build --network=host -t cann-bench:cann9.0.0-910b-latest \
#       --build-arg CANN_VERSION=9.0.0 --build-arg DEVICE=910b .
#
#   docker build --network=host -t cann-bench:cann9.0.0-950-latest \
#       --build-arg CANN_VERSION=9.0.0 --build-arg DEVICE=950 .
#
# 可用 DEVICE 值：910b / 910c / 950 等（对应 AscendHub 镜像 tag 中的硬件段）
# 完整镜像 tag 格式：cann:<CANN_VERSION>-<DEVICE>-ubuntu22.04-py3.12

ARG CANN_VERSION=9.0.0
ARG DEVICE=910b

FROM swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:${CANN_VERSION}-${DEVICE}-ubuntu22.04-py3.12

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl git ca-certificates build-essential cmake \
    clang lld ninja-build \
    tmux clangd gh rsync && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN curl -LSf https://astral.sh/uv/install.sh | sh && \
    install -m 0755 /root/.local/bin/uv  /usr/local/bin/uv && \
    install -m 0755 /root/.local/bin/uvx /usr/local/bin/uvx

ARG PYPI_INDEX_URL=
ENV UV_INDEX_URL=${PYPI_INDEX_URL}

# torch_npu install needs these at build time to locate driver libs.
ENV ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
ENV LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:${ASCEND_HOME_PATH}/aarch64-linux/lib64

# 使用华为云源（清华源持续 403）；PYPI_INDEX_URL build-arg 可覆盖此默认源。
RUN uv pip install --system --index-url ${PYPI_INDEX_URL:-https://mirrors.huaweicloud.com/repository/pypi/simple/} \
    filelock typing-extensions sympy networkx jinja2 fsspec mpmath markupsafe \
    numpy pyyaml decorator scipy attrs cloudpickle \
    ml-dtypes psutil tornado pandas expecttest protobuf \
    pytest pytest-timeout 'ruamel.yaml>=0.18' build setuptools wheel
RUN uv pip install --system --no-deps --index-url https://download.pytorch.org/whl/cpu torch==2.9.0
RUN uv pip install --system --no-deps --index-url ${PYPI_INDEX_URL:-https://mirrors.huaweicloud.com/repository/pypi/simple/} torch_npu==2.9.0.post2

# Pre-populate uid 1001 in /etc/passwd so getpass/getuser resolves in the
# sandbox runtime (which executes processes as uid 1001); not a privilege drop.
RUN echo "benchuser:x:1001:1001:bench user:/tmp:/bin/bash" >> /etc/passwd

COPY test_env.py    /test_env.py
COPY entrypoint.sh  /entrypoint.sh
RUN chmod +x /entrypoint.sh /test_env.py

# Drop the mirror ENV so runtime stays clean (build-time only).
ENV UV_INDEX_URL=

ENTRYPOINT ["/entrypoint.sh"]
WORKDIR /workspace
