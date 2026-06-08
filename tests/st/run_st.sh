#!/bin/bash
# Golden-candidate ST entry: pytest over the golden ops on the NPU (CI ST stage:
# `bash tests/st/run_st.sh`; tests/st/run_st_docker.sh wraps the docker run). Args pass to
# pytest; default selection (Cummin smoke unless --full / -k / -m) lives in tests/st/conftest.py.
set -uo pipefail   # not -e: run every case + still emit junit/reports when some fail

# ST-owned (gitignored) output, never the project's shared build/ — so wiping it can't
# touch the build tree. On the bind-mounted /workspace, so it outlives the transient container.
ST_OUT="${ST_OUT:-tests/st/_artifacts}"
rm -rf "$ST_OUT"
mkdir -p "$ST_OUT"
export PYTHONUNBUFFERED=1

# basetemp under $ST_OUT (bind mount), not the container's /tmp: if the container dies
# mid-run, the candidate/trimmed-tree/report tmp survives for post-mortem (cleaned on normal exit).
PYTEST_ARGS=(tests/st/test_golden_npu_mock.py "$@" -v -ra -p no:cacheprovider \
             --junitxml="$ST_OUT/matrix_junit.xml" --basetemp="$ST_OUT/tmp")

python -m pytest "${PYTEST_ARGS[@]}"
rc=$?
# single-run 集成口径:整个选中子集只产一份 eval_*.{json,md,html}(含全部算子)。把这套矩阵级
# 报告收进 $ST_OUT/ 根(python helper);bulky 的 prof_data/trace/msprof 丢弃。
# 路径经环境变量传入,不内插进 python -c 字符串(否则恶意 ST_OUT 可注入任意代码)。
n=$(PYTHONPATH=tests/st ST_TMP="$ST_OUT/tmp" ST_OUT_DIR="$ST_OUT" python3 -c \
  "import os; from harness.report import collect_artifacts as c; print(c(os.environ['ST_TMP'], os.environ['ST_OUT_DIR']))" \
  2>/dev/null || echo 0)
rm -rf "$ST_OUT/tmp"
echo "PYTEST_RC=${rc}  artifacts: $ST_OUT (${n} report)"
exit "${rc}"
