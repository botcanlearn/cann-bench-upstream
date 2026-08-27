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
ARG BASE_IMAGE=cann-toolkit-base:9.1.0-py3.13

FROM ${BASE_IMAGE}

ARG PYPI_MIRROR=
ARG TORCH_MIRROR=
ARG UV_PYTHON_INSTALL_MIRROR=

# This layer is where the per-SoC fork lives; everything architecture-shaped and every version-shaped
# thing already happened in the base, and arrives as the inherited ENV CANN_ARCH / CANN_VERSION
# (deliberately NOT re-declared here -- local knobs could disagree with the base and quietly install
# the wrong architecture's, or the wrong release's, ops on top of its toolkit).
# NPU_ARCH picks the SoC cann_bench_utils compiles kernels for; OPS_PKG names that SoC's ops .run
# (spelled differently from the compiler flag: ascend910b <-> 910b, ascend950 <-> 950).
# docker/eval/build.sh derives NPU_ARCH/OPS_PKG/OPS_MODE for you; set them by hand only for a one-off.
ARG NPU_ARCH=ascend910b
ARG OPS_PKG=910b

# WHY this exists at all: a submission "cheats" by CALLING a builtin (aclnn<Op> / torch_npu op), which
# dispatches into the multi-GB tbe/kernel BINARY tree (4.2G on 910B, 5.0G on 950). The ~151M tbe/impl
# AscendC SOURCE tree is not a cheat -- it is a legitimate reference we may want an agent to read.
#   none     : never install ops -- the base is already ops-less, so this costs nothing. Blocks builtins
#              because libopapi (the aclnn entry library, shipped in ops) is absent; the toolkit's own
#              small ~11M kernel tree is not enough to launch one. opp/ still exists, including the empty
#              opp/vendors that .run-form custom-op submissions install into.
#   refonly  : install ops, then strip ONLY the kernel binaries IN THE SAME LAYER -> keep the impl source
#              as reference; libopapi is present but every builtin fails to launch.
#   full     : install ops and keep everything. Cheatable; for re-collecting aclnn baselines.
#
# WHICH ONE: `none` is right on 910B and WRONG on 950. Measured on a 950PR: with no ops at all even a
# bare `torch.arange(8).npu()` dies -- the H2D copy routes through aclnnInplaceCopy there (ERR01007
# "OPS feature not supported"), where 910B does a plain aclrtMemcpy and needs nothing. `refonly` fixes
# that on 950 and still blocks builtins (matmul dies 561103 "Parse dynamic kernel config fail"), so 950
# keeps the full anti-cheat posture -- it just cannot use `none`. The guard below refuses that combo
# rather than shipping an image that dies on its first tensor.
ARG OPS_MODE=none
# Empty = derive from the inherited CANN_VERSION + CANN_ARCH below. Set it only to pin an exact .run
# (the ops naming has held across releases -- 9.1.0 is published for 910b/950 x aarch64/x86_64 under
# exactly this scheme -- so this is a pin, not a portability crutch).
ARG CANN_OPS_URL=
ARG OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp

# The ops tree is GBs and never changes -- keep it as the first (bottom-most) layer so the python /
# harness layers below rebuild independently of it.
RUN set -e; \
    case "${OPS_MODE}" in none|refonly|full) ;; *) echo "invalid OPS_MODE=${OPS_MODE}" >&2; exit 1;; esac; \
    : "${CANN_ARCH:?base image did not set CANN_ARCH -- rebuild docker/base}"; \
    : "${CANN_VERSION:?base image did not set CANN_VERSION -- rebuild docker/base}"; \
    if [ "$(uname -m)" != "${CANN_ARCH}" ]; then \
      echo "base image is ${CANN_ARCH} but this is a $(uname -m) host -- native builds only." >&2; \
      exit 1; \
    fi; \
    if [ "${NPU_ARCH}" = ascend950 ] && [ "${OPS_MODE}" = none ]; then \
      echo "OPS_MODE=none is not usable on ${NPU_ARCH}: the H2D copy needs aclnnInplaceCopy (ERR01007)." >&2; \
      echo "Use OPS_MODE=refonly -- it keeps the anti-cheat posture on 950. See the note above." >&2; \
      exit 1; \
    fi; \
    OPS_URL="${CANN_OPS_URL:-https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%20${CANN_VERSION}/Ascend-cann-${OPS_PKG}-ops_${CANN_VERSION}_linux-${CANN_ARCH}.run}"; \
    if [ "${OPS_MODE}" != none ]; then \
      wget --quiet --header="Referer: https://www.hiascend.com/" "${OPS_URL}" -O /tmp/ops.run \
      && chmod +x /tmp/ops.run && /tmp/ops.run --quiet --install --install-for-all && rm -f /tmp/ops.run; \
    fi; \
    if [ "${OPS_MODE}" = refonly ]; then rm -rf "${OPP_PATH}/built-in/op_impl/ai_core/tbe/kernel"; fi; \
    echo "OPS_MODE=${OPS_MODE} NPU_ARCH=${NPU_ARCH} CANN_ARCH=${CANN_ARCH} CANN_VERSION=${CANN_VERSION}"
ENV OPS_MODE=${OPS_MODE}

# kernel_eval imports more than the direct-launch base needed (pandas / ruamel.yaml / protobuf /
# torchvision goldens / ...). docker/eval/pyproject.toml is a STRICT SUPERSET of the base's, so this
# `uv sync` only ADDS -- it reconciles the same /opt/venv without uninstalling the base's torch.
# Mirror handling is the base's trick: rewrite the lock's canonical URLs in place, same hashes, so
# --frozen still verifies and mirroring costs no reproducibility.
#
# UV_INDEX_URL is needed ON TOP of that rewrite, and only here. `en-dtypes` is the one entry in this
# lock published as an sdist with no wheel, so uv must BUILD it -- and building means resolving its
# build-system.requires (setuptools, numpy) from an INDEX, which the URL rewrite does not touch. Left
# unset that resolution goes to pypi.org and times out in-region ("Failed to fetch
# https://pypi.org/simple/setuptools/ ... operation timed out"), failing the whole layer after ~4min.
# docker/base's lock has no sdist-only entry, which is why it never needed this.
COPY docker/eval/pyproject.toml docker/eval/uv.lock /opt/eval-env/
RUN L=/opt/eval-env/uv.lock; \
    if [ -n "${PYPI_MIRROR}" ];  then sed -i "s|https://files.pythonhosted.org/packages/|${PYPI_MIRROR%/}/packages/|g" "$L"; \
                                      export UV_INDEX_URL="${PYPI_MIRROR%/}/simple"; fi; \
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
# It embeds SoC-specific kernels (NPU_ARCH, declared with the other knobs at the top) -- which is what
# makes this image per-SoC even though the base is not. Verified to build and run for both ascend910b
# and ascend950.
RUN . /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true; \
    cd /opt/cann-bench/src/cann_bench_utils \
    && PYTHON=/opt/venv/bin/python3 bash build.sh --clean --soc=${NPU_ARCH} \
    && /opt/venv/bin/python3 -m pip install --no-deps --force-reinstall dist/cann_bench_utils-*.whl \
    && /opt/venv/bin/python3 -c 'from cann_bench_utils import cann_bench_warmup, cann_bench_cache_clean'
ENV NPU_ARCH=${NPU_ARCH}

# Which image produced a report. docker/eval/build.sh passes its own tag here and setup_info reads it
# for the report's "docker" field -- that field used to be the hardcoded string "cake-ci / CANN 9.0.0",
# which kept appearing long after both the image and the CANN version had moved on. Empty on a
# hand-rolled `docker build`; setup_info then says "container" rather than inventing a name.
ARG CANN_BENCH_IMAGE=
ENV CANN_BENCH_IMAGE=${CANN_BENCH_IMAGE}

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
