# cann-toolkit-base: minimal CANN toolkit + torch_npu direct-launch base. Single self-contained Dockerfile.
#
# WHY toolkit-only (no ops/nnal): direct-launch (bisheng-compiled AscendC/CCE kernels launched
# via `<<<>>>`) never calls aclnn, so ops (libopapi + kernel binaries) and nnal (ATB/SiP) are dead
# weight. Verified on 910b: torch_npu imports + npu memcpy works with libopapi absent.
# Pairing (Ascend/pytorch COMPATIBILITY.md): torch_npu 2.10.0.post4 <-> torch 2.10.0 <-> CANN 9.1.0, py3.13.
# The torch_npu 2.10 line pairs ONLY with 9.1.0 -- that table has no 9.0.1 row at all, and the
# 2.10.0.post2 this image used to pin appears under no CANN version. Move the two together or you
# land off-table again.

# ----------------------------------------------------------------------------------------------------------
# Mirror / version knobs -- every one DEFAULTS to the official/global source; pass --build-arg to swap to an
# in-region mirror on network-restricted hosts. Full CN-mirror recipe:
#   --build-arg BASE_OS=docker.m.daocloud.io/library/debian:12-slim  (base image; docker.io blocked)
#   --build-arg APT_MIRROR=mirrors.huaweicloud.com                   (apt; deb.debian.org throttled)
#   --build-arg UV_IMAGE=ghcr.m.daocloud.io/astral-sh/uv:0.11.29      (uv binary; ghcr slow)
#   --build-arg UV_PYTHON_INSTALL_MIRROR=https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone
#                                                                    (uv's cpython; github flaky. USTC keeps
#                                                                     LatestRelease only -> no pinned tag)
#   --build-arg PYPI_MIRROR=https://mirrors.huaweicloud.com/repository/pypi   (general wheels)
#   --build-arg TORCH_MIRROR=https://mirror.nju.edu.cn/pytorch/whl/cpu    (torch +cpu wheel)
# PYPI_MIRROR/TORCH_MIRROR rewrite the hash-pinned lock's canonical URLs IN PLACE (same hashes, so `uv sync
# --frozen` still verifies) -> mirroring keeps reproducibility. debian:12-slim proven with CANN by triton-
# ascend. UV_IMAGE needs a global ARG + `FROM ... AS uvbin` (COPY --from can't expand a build-arg image ref).
# ----------------------------------------------------------------------------------------------------------
ARG BASE_OS=debian:12-slim
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.29
ARG APT_MIRROR=
ARG UV_PYTHON_INSTALL_MIRROR=
ARG PYPI_MIRROR=
ARG TORCH_MIRROR=
ARG CANN_VERSION=9.1.0
# CANN_VERSION alone is enough to move versions -- the toolkit .run and its <arch> variants follow one
# naming scheme on ascend-repo, verified present for 9.1.0 in both architectures. CANN_TOOLKIT_URL is
# the escape hatch for the releases that do not: some land on the ascend-cann-open bucket instead
# (ascend-cann-open.obs.cn-north-4...), and some are published as a combined `Ascend-cann_<v>_linux-
# <arch>.run` rather than `Ascend-cann-toolkit_...`. Empty = derive from CANN_VERSION + ARCH.
ARG CANN_TOOLKIT_URL=

FROM ${UV_IMAGE} AS uvbin
FROM ${BASE_OS}

# Docker resets ARG scope at each FROM -- re-declare the ones the build stage below uses.
ARG APT_MIRROR
ARG UV_PYTHON_INSTALL_MIRROR
ARG PYPI_MIRROR
ARG TORCH_MIRROR
ARG CANN_VERSION
ARG CANN_TOOLKIT_URL

# ARCH selects the CPU architecture of every arch-shaped path: the toolkit .run, the toolkit's
# <arch>-linux/ tree, and the wheels uv picks. Build it NATIVELY on a host of that architecture --
# this is not a cross-compile knob, and there is no --platform anywhere.
#
# It was hardcoded aarch64 until x86_64 was actually exercised (Ascend hosts are usually Kunpeng/ARM,
# and the old comment rightly refused a knob nobody built). Both are now validated end to end:
# aarch64/910B2 and x86_64/950PR each install the toolkit, `uv sync --frozen` off the same lock (it
# already carries both architectures' wheels, torch_npu included), stub libhccl, build cann_bench_utils,
# and see their card. The other half of that old objection -- "a JSON ENTRYPOINT can't expand
# build-args" -- is answered below by baking the arch into a generated env script instead.
ARG ARCH=aarch64

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# plugin compiles with g++, kernel with bisheng. libsqlite3-0: msprof's profiling analysis
# (the kernel_details.csv export the kernel_eval perf stage triggers via torch_npu.profiler) does
# `import sqlite3`, whose _sqlite3 ext dynlinks libsqlite3.so.0 -- debian-slim omits it, so without
# it perf yields no kernel timing (accuracy is unaffected). APT_MIRROR (top): sed-swap deb.debian.org in
# the deb822 sources (bookworm /etc/apt/sources.list.d/debian.sources) to an in-region mirror.
RUN if [ -n "${APT_MIRROR}" ]; then sed -i "s|deb.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list.d/debian.sources; fi \
    && apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl wget git \
    build-essential g++ cmake ninja-build patchelf pkg-config xz-utils \
    libsqlite3-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# uv-managed standalone py3.13 -- its libpython3.13.so satisfies CANN; system python isn't 3.13.
COPY --from=uvbin /uv /uvx /usr/local/bin/
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
ENV UV_PROJECT_ENVIRONMENT=/opt/venv

# Lockfile-driven env: `uv sync --frozen` installs the exact hash-pinned graph (pyproject + uv.lock)
# into /opt/venv. WHY pip/setuptools/wheel are in the lock: the Toolkit ge-compiler sub-installer runs
# `pip3 install`, and the submitted extension's setup.py / `python -m build` needs setuptools.
# UV_PYTHON_INSTALL_MIRROR redirects uv's cpython download off github; PYPI_MIRROR/TORCH_MIRROR rewrite the
# lock's canonical wheel URLs to a mirror IN PLACE (same hashes -> --frozen still verifies). All "" = official.
COPY pyproject.toml uv.lock /opt/base-env/
RUN L=/opt/base-env/uv.lock; \
    if [ -n "${PYPI_MIRROR}" ];  then sed -i "s|https://files.pythonhosted.org/packages/|${PYPI_MIRROR%/}/packages/|g" "$L"; fi; \
    if [ -n "${TORCH_MIRROR}" ]; then sed -i "s|https://download-r2.pytorch.org/whl/cpu/|${TORCH_MIRROR%/}/|g" "$L"; fi; \
    UV_PYTHON_INSTALL_MIRROR="${UV_PYTHON_INSTALL_MIRROR}" uv sync --frozen --project /opt/base-env
ENV PATH=/opt/venv/bin:${PATH}

# Toolkit ONLY -- the ops/nnal .run the upstream cann-container-image also fetches are omitted.
RUN cd /tmp \
    && URL="${CANN_TOOLKIT_URL:-https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%20${CANN_VERSION}/Ascend-cann-toolkit_${CANN_VERSION}_linux-${ARCH}.run}" \
    && echo "toolkit: ${URL}" \
    && wget --quiet --header="Referer: https://www.hiascend.com/" -O toolkit.run "${URL}" \
    && chmod +x toolkit.run && ./toolkit.run --quiet --install --install-for-all && rm -f toolkit.run

# WHY stub: torch_npu _C.so has an unconditional DT_NEEDED on libhccl.so but imports 0 symbols from it
# -- an empty SONAME-only .so satisfies the loader without the real HCCL lib (which ships in ops).
# Written through the installer's own `latest` symlink rather than the versioned cann-<v>/ directory it
# points at: same inode, but it does not assume the installer keeps naming that directory after
# CANN_VERSION, and it is the same path LD_LIBRARY_PATH below resolves.
RUN printf 'int __hccl_stub;\n' | gcc -shared -fPIC -x c -Wl,-soname,libhccl.so \
    -o /usr/local/Ascend/ascend-toolkit/latest/${ARCH}-linux/lib64/libhccl.so -

# ONE generated env script is the single source of truth for the Ascend environment, and it is where
# ARCH gets baked in -- a JSON ENTRYPOINT cannot expand a build-arg, but a script it sources can carry
# one. Three entry paths all funnel through it: ENTRYPOINT (`docker run img <cmd>`), /etc/bash.bashrc
# (`docker exec`), /etc/profile.d (login shells, e.g. a runner that does `bash -lc`, whose /etc/profile
# would otherwise reset PATH and drop /opt/venv/bin). The sentinel keeps nested shells from stacking
# PATH entries. LD_LIBRARY_PATH needs <arch>-linux/lib64 explicitly: set_env.sh omits it, but the
# stubbed libhccl and its siblings live there.
ENV ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
ENV PATH=/opt/venv/bin:${ASCEND_HOME_PATH}/${ARCH}-linux/ccec_compiler/bin:${PATH}
# Architecture and CANN version are properties of THIS image, so publish them for anything built FROM
# here (docker/eval needs both to pick the right ops .run). A child that re-declared its own ARG could
# disagree with the base and silently install the wrong architecture's -- or the wrong release's --
# ops on top of this toolkit; inheriting makes that impossible.
ENV CANN_ARCH=${ARCH}
ENV CANN_VERSION=${CANN_VERSION}
RUN printf '%s\n' \
    '# Only set_env.sh is guarded: it appends unboundedly, so re-sourcing it in nested shells' \
    '# would grow the environment without limit. The exported vars it defines survive anyway.' \
    'if [ -z "${_CANN_ENV_DONE:-}" ]; then' \
    '  export _CANN_ENV_DONE=1' \
    '  source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true' \
    'fi' \
    '# PATH / LD_LIBRARY_PATH are re-asserted EVERY time, membership-tested rather than' \
    '# guard-tested. Anything downstream may reset PATH after the guard is already exported --' \
    "# /etc/profile in a login shell does exactly that -- and a one-shot guard would then skip" \
    '# the repair, leaving the shell without /opt/venv/bin (python3 disappears). The case test' \
    '# keeps repeated sourcing from growing them.' \
    'case ":${PATH}:" in' \
    '  *":/opt/venv/bin:"*) ;;' \
    "  *) export PATH=/opt/venv/bin:${ASCEND_HOME_PATH}/${ARCH}-linux/ccec_compiler/bin:\${PATH} ;;" \
    'esac' \
    "case \":\${LD_LIBRARY_PATH:-}:\" in" \
    "  *\":${ASCEND_HOME_PATH}/${ARCH}-linux/lib64:\"*) ;;" \
    "  *) export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/${ARCH}-linux/lib64:\${LD_LIBRARY_PATH} ;;" \
    'esac' \
    > /etc/cann-env.sh \
 && chmod 0644 /etc/cann-env.sh \
 && ln -sf /etc/cann-env.sh /etc/profile.d/10-cann.sh \
 && printf 'source /etc/cann-env.sh\n' >> /etc/bash.bashrc
ENV BASH_ENV=/etc/cann-env.sh

ENTRYPOINT ["/bin/bash", "-c", "source /etc/cann-env.sh; exec \"$@\"", "bash"]
WORKDIR /workspace
CMD ["bash"]
