#!/bin/bash
# Golden-candidate ST entry: pytest over the golden ops on the NPU. CI ST stage runs this
# directly inside its NPU image(`bash tests/st/run_st.sh`)。Args pass to pytest;
# default selection (unless --full / -k / -m) lives in tests/st/conftest.py.
set -uo pipefail   # not -e: run every case + still emit junit/reports when some fail

# CI keeps this job's stdout and nothing else — no artifact download, no container to re-enter.
# So every fact needed to localise a failure has to be printed here. Sections are tagged [ST]
# / [ST-DIAG] to stay greppable inside the pipeline's interleaved, timestamp-prefixed stream.
banner() { echo ""; echo "[ST] ===== $* ====="; }

# One greppable line per run, so a red/green history can be correlated against WHERE it ran.
# Host identity only — the card is right above in the full npu-smi table, no point re-parsing it.
# boot_epoch (wall clock - /proc/uptime, which inside this container is the HOST's) is stable
# per physical machine across runs; nothing else here is — the pod name is fresh every run.
fingerprint() {
  local up boot mem
  up=$(awk '{printf "%.0f", $1}' /proc/uptime 2>/dev/null)
  # No uptime -> no boot epoch. Printing `now` here would look like a real (and always
  # different) host id, which is worse than admitting we don't know.
  boot=$([ -n "$up" ] && echo $(( $(date +%s) - up )) || echo n/a)
  mem=$(awk '/^MemTotal/{printf "%.0fG", $2/1048576}' /proc/meminfo 2>/dev/null)
  echo "[ST] fingerprint host=$(hostname 2>/dev/null) nproc=$(nproc 2>/dev/null || echo n/a)" \
       "memtotal=${mem:-n/a} uptime_s=${up:-n/a} boot_epoch=${boot}"
}

banner "run context"
date -u '+%Y-%m-%dT%H:%M:%SZ  (UTC)'
echo "host=$(hostname 2>/dev/null)  pwd=$(pwd)  user=$(id -un 2>/dev/null)"
echo "git=$(git rev-parse --short HEAD 2>/dev/null || echo n/a)  args=$*"
env | grep -E '^(ASCEND|ATB|ACL|LD_LIBRARY_PATH|PYTHONPATH|ST_|PR_FILELIST)' | sort || true

banner "device"
npu-smi info
fingerprint

banner "toolchain / resources"
# Versions + headroom, not decoration: a golden case can materialise GBs of reference tensors,
# and a full disk or a stale torch/torch_npu ABI both surface downstream as an opaque rc=1.
python -c 'import sys; print("python", sys.version.replace("\n", " "), sys.executable)' || true
python -c 'import torch, torch_npu; print("torch", torch.__version__, "torch_npu", torch_npu.__version__)' 2>&1 | tail -3 || true
cat "${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}/version.cfg" 2>/dev/null || echo "CANN version.cfg: n/a"
df -h . /tmp "${TMPDIR:-/tmp}" 2>/dev/null | sort -u || true
free -g 2>/dev/null || true

# make sure ST_OUT is clean
ST_OUT="${ST_OUT:-tests/st/_artifacts}"
rm -rf "$ST_OUT"
mkdir -p "$ST_OUT"
export ST_OUT                 # harness.eval_run writes the kernel_eval cli log here
export PYTHONUNBUFFERED=1

PYTEST_ARGS=(tests/st/test_golden_npu_mock.py)
# 无参 `bash tests/st/run_st.sh` 是 CI 入口, 应提供改动清单 $PR_FILELIST
PR_FILELIST="${PR_FILELIST:-pr_filelist.txt}"
if [ "$#" -eq 0 ]; then
  if [ ! -r "$PR_FILELIST" ]; then
    echo "[run_st] ERROR: 未找到改动清单 $PR_FILELIST, CI 应在仓库根产出该清单." >&2
    echo "[run_st]        本地手动跑请显式传 -k/-m/--full (如 bash tests/st/run_st.sh -k Cummin)." >&2
    exit 2
  fi
  expr=$(PYTHONPATH=tests/st python -m harness.select_from_changes "$PR_FILELIST" 2>/dev/null || true)
  if [ -n "$expr" ]; then
    echo "[run_st] $PR_FILELIST → -k '$expr'"
    PYTEST_ARGS+=(-k "$expr")
  else
    echo "[run_st] $PR_FILELIST 无 tasks/ 算子改动 → 默认组 (见 tests/st/conftest.py)"
  fi
fi
# basetemp under $ST_OUT (bind mount), not the container's /tmp: if the container dies
# mid-run, the candidate/trimmed-tree/report tmp survives for post-mortem (cleaned on normal exit).
PYTEST_ARGS+=("$@" -v -ra -p no:cacheprovider \
              --junitxml="$ST_OUT/matrix_junit.xml" --basetemp="$ST_OUT/tmp")

echo ""
echo "################################  CANN-BENCH ST  ################################"
python -m pytest "${PYTEST_ARGS[@]}"
rc=$?

banner "device after run"
npu-smi info                     # post-run health: a case that took the device down shows here
df -h . 2>/dev/null || true      # golden references are GB-scale; ENOSPC reads as a bare rc=1

# single-run 集成口径: 整个选中子集只产一份 eval_*.{json,md,html}(含全部算子)。
# 路径经环境变量传入,不内插进 python -c 字符串 (否则恶意 ST_OUT 可注入任意代码)。
# stderr 不再丢弃: 收集失败本身就是 ST 出错位置的一部分。
n=$(PYTHONPATH=tests/st ST_TMP="$ST_OUT/tmp" ST_OUT_DIR="$ST_OUT" python3 -c \
  "import os; from harness.report import collect_artifacts as c; print(c(os.environ['ST_TMP'], os.environ['ST_OUT_DIR']))" \
  || { echo "[ST] collect_artifacts 失败 (见上方 traceback)" >&2; echo 0; })
rm -rf "$ST_OUT/tmp"

# Post-mortem: report + cli log are on disk under $ST_OUT, but CI ships only stdout — print a
# bounded digest of them. Failure only; a green run stays quiet.
if [ "$rc" -ne 0 ]; then
  PYTHONPATH=tests/st python3 -m harness.diagnose "$ST_OUT" \
    || echo "[ST] diagnose 自身失败 (见上方 traceback)" >&2
fi

echo "################################################################################"
if [ "$rc" -eq 0 ]; then
  echo "##  CANN-BENCH ST PASSED (rc=0) -- artifacts: $ST_OUT (${n} report)"
else
  echo "##  CANN-BENCH ST FAILED (rc=${rc}) -- 见上方 pytest 输出 -- artifacts: $ST_OUT (${n} report)"
fi
echo "PYTEST_RC=${rc}  artifacts: $ST_OUT (${n} report)"   # 机器可解析行,保持不变
echo "################################################################################"
exit "${rc}"
