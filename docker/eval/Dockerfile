# cann-bench-eval -- the benchmark IS the image. `docker run <image> [源码目录] [选项]` runs a full
# kernel_eval evaluation; nothing outside the image is needed but the NPU driver and the submission.
#
# WHY: until now an evaluation needed a live cann-bench worktree plus a hand-assembled CANN/torch_npu
# environment -- scripts/run_evaluation.sh derives PROJECT_ROOT from its own location and writes
# reports back into the repo. That makes a run depend on binaries that get copied from host to host,
# and makes "which benchmark version produced this score" unanswerable. Here the harness
# (src/kernel_eval + tasks/ + the cann_bench_utils warmup provider) is frozen into the image, so the
# image tag IS the benchmark version; only /submission and /reports are mounted.
#
# BUILD CONTEXT IS THE REPO ROOT (this Dockerfile COPYs src/ tasks/ scripts/) -- use docker/eval/build.sh,
# or `docker build -f docker/eval/Dockerfile .` from the repo root. Building from inside docker/eval/ fails.

# ----------------------------------------------------------------------------------------------------------
# Knobs. Every mirror arg defaults to the official/global source, exactly like docker/base; pass
# --build-arg to swap to an in-region mirror on network-restricted hosts (see README).
# ----------------------------------------------------------------------------------------------------------
ARG BASE_IMAGE=cann-toolkit-base:9.0.1-py3.13

FROM ${BASE_IMAGE}

ARG PYPI_MIRROR=
ARG TORCH_MIRROR=
ARG UV_PYTHON_INSTALL_MIRROR=
ARG CANN_VERSION=9.0.1
ARG ARCH=aarch64

# WHY this exists at all: a submission "cheats" by CALLING a builtin (aclnn<Op> / torch_npu op), which
# dispatches into the 4.2G tbe/kernel BINARY tree. The 151M tbe/impl AscendC SOURCE tree is not a cheat
# -- it is a legitimate reference we may want an agent to read.
#   none    (default): never install ops. The base is already 0-ops, so this costs nothing and blocks
#                      every builtin. opp/ still exists (the toolkit ships it, incl. the empty
#                      opp/vendors that .run-form custom-op submissions install into).
#   refonly          : install ops, then strip ONLY the kernel binaries IN THE SAME LAYER -> keep the
#                      impl source as reference, builtins fail to launch. Use this if a run needs opp
#                      machinery that 0-ops lacks (see README's perf note).
#   full             : install ops and keep everything. Cheatable; for re-collecting aclnn baselines.
ARG OPS_MODE=none
ARG CANN_OPS_URL=https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%20${CANN_VERSION}/Ascend-cann-910b-ops_${CANN_VERSION}_linux-${ARCH}.run
ARG OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp

# The ops tree is GBs and never changes -- keep it as the first (bottom-most) layer so the python /
# harness layers below rebuild independently of it.
RUN set -e; \
    case "${OPS_MODE}" in none|refonly|full) ;; *) echo "invalid OPS_MODE=${OPS_MODE}" >&2; exit 1;; esac; \
    if [ "${OPS_MODE}" != none ]; then \
      wget --quiet --header="Referer: https://www.hiascend.com/" "${CANN_OPS_URL}" -O /tmp/ops.run \
      && chmod +x /tmp/ops.run && /tmp/ops.run --quiet --install --install-for-all && rm -f /tmp/ops.run; \
    fi; \
    if [ "${OPS_MODE}" = refonly ]; then rm -rf "${OPP_PATH}/built-in/op_impl/ai_core/tbe/kernel"; fi; \
    echo "OPS_MODE=${OPS_MODE}"
ENV OPS_MODE=${OPS_MODE}

# kernel_eval imports more than the direct-launch base needed (pandas / ruamel.yaml / protobuf /
# torchvision goldens / ...). docker/eval/pyproject.toml is a STRICT SUPERSET of the base's, so this
# `uv sync` only ADDS -- it reconciles the same /opt/venv without uninstalling the base's torch.
# Mirror handling is the base's trick: rewrite the lock's canonical URLs in place, same hashes, so
# --frozen still verifies and mirroring costs no reproducibility.
COPY docker/eval/pyproject.toml docker/eval/uv.lock /opt/eval-env/
RUN L=/opt/eval-env/uv.lock; \
    if [ -n "${PYPI_MIRROR}" ];  then sed -i "s|https://files.pythonhosted.org/packages/|${PYPI_MIRROR%/}/packages/|g" "$L"; fi; \
    if [ -n "${TORCH_MIRROR}" ]; then sed -i "s|https://download-r2.pytorch.org/whl/cpu/|${TORCH_MIRROR%/}/|g" "$L"; fi; \
    UV_PYTHON_INSTALL_MIRROR="${UV_PYTHON_INSTALL_MIRROR}" uv sync --frozen --project /opt/eval-env

# The frozen harness. tasks/ comes along on purpose: it carries the golden references and the case
# matrix, so baking it is what makes the image tag mean a benchmark version.
COPY src     /opt/cann-bench/src
COPY tasks   /opt/cann-bench/tasks
COPY scripts /opt/cann-bench/scripts
COPY pyproject.toml VERSION /opt/cann-bench/
ENV CANN_BENCH_DIR=/opt/cann-bench

# cann_bench_utils is a HARD dependency of every evaluation (the V3 anti-cheat warmup / L2-cache-clean
# provider): with the builtin kernel tree unavailable the framework cannot use torch.matmul / torch.max
# for the frequency ramp, so run_evaluation.sh's ensure_cann_bench_utils() would try to compile it at
# RUN time. Compile and install it here instead -- it needs only bisheng, no NPU -- so the container
# starts straight into evaluating and the wheel is part of the frozen artifact.
# It embeds SoC-specific kernels, hence NPU_ARCH: this image is per-SoC even though the base is not.
ARG NPU_ARCH=ascend910b
RUN . /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true; \
    cd /opt/cann-bench/src/cann_bench_utils \
    && PYTHON=/opt/venv/bin/python3 bash build.sh --clean --soc=${NPU_ARCH} \
    && /opt/venv/bin/python3 -m pip install --no-deps --force-reinstall dist/cann_bench_utils-*.whl \
    && /opt/venv/bin/python3 -c 'from cann_bench_utils import cann_bench_warmup, cann_bench_cache_clean'
ENV NPU_ARCH=${NPU_ARCH}

# Optional Triton-Ascend runtime (large); same semantics as the dev image's build-arg.
ARG TRITON_ASCEND_VERSION=
ARG TRITON_ASCEND_INDEX_URL=https://triton-ascend.osinfra.cn/pypi/simple
RUN if [ -n "${TRITON_ASCEND_VERSION}" ]; then \
      uv pip install --python /opt/venv/bin/python3 \
        ${PYPI_MIRROR:+--index-url ${PYPI_MIRROR}/simple} \
        --extra-index-url "${TRITON_ASCEND_INDEX_URL}" \
        "triton-ascend==${TRITON_ASCEND_VERSION}"; \
    fi
ENV TRITON_ASCEND_VERSION=${TRITON_ASCEND_VERSION}
ENV TRITON_CACHE_DIR=/tmp/cann-bench-triton-cache

# Mount points. /submission is read as-is (the entrypoint copies it before building, so :ro is fine);
# /reports is the only thing the run writes. Both are pre-created so a bare `docker run` without -v
# still has somewhere to go instead of failing on a missing path.
ENV SUBMISSION_DIR=/submission
ENV REPORTS_DIR=/reports
RUN mkdir -p /submission /reports /work ${TRITON_CACHE_DIR} && chmod 1777 /reports /work ${TRITON_CACHE_DIR}

COPY docker/eval/entrypoint.sh /entrypoint.sh
COPY docker/eval/self_test.py  /opt/cann-bench/docker-self-test.py
RUN chmod +x /entrypoint.sh

# Build-time-only mirror env must not leak into the run.
ENV UV_INDEX_URL=

WORKDIR /work
ENTRYPOINT ["/entrypoint.sh"]
