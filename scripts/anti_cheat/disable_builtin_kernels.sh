#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------
# Benchmark-integrity: disable the built-in AiCore *operator kernel binaries* for every benchmarked op,
# so a submission cannot "cheat" by launching the stock kernel (via aclnn<Op> / ADD_TO_LAUNCHER_LIST_AICORE)
# instead of providing its own kernel.
#
# Only the prebuilt kernel binary dir (kernel/<soc>/<cat>/<op>/  -> *.o + *.json) is moved out of OPP.
# The TBE/AscendC impl sources, op_proto and registration are LEFT INTACT, and torch_npu is unaffected.
# Device-side AscendC intrinsics (AscendC::Add/Mul/Exp/Mmad/...) compile into the candidate kernel and are
# NOT affected by removing operator binaries.
#
# PROTECTED (never listed): MatMul*/ReduceMax* (used by perf_eval freq-boost / L2-flush) + generic primitives.
#
# Implemented with `mv` (NOT `rm`): each kernel dir is MOVED into the backup dir — that single step both
# backs it up and removes it from OPP, so no data is ever deleted. Run restore_builtin_kernels.sh to undo.
#
# Usage:
#   bash scripts/anti_cheat/disable_builtin_kernels.sh [--soc=ascend910b] [--list=<file>] [--backup-dir=<dir>] [--dry-run]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOC="ascend910b"
LIST="${SCRIPT_DIR}/benchmarked_kernels.txt"
BACKUP_DIR="${CANN_BENCH_KERNEL_BACKUP:-$HOME/.cann_bench_kernel_backup}"
DRY_RUN=0
ASSUME_YES=0
for arg in "$@"; do
  case "$arg" in
    --soc=*)        SOC="${arg#*=}";;
    --list=*)       LIST="${arg#*=}";;
    --backup-dir=*) BACKUP_DIR="${arg#*=}";;
    --dry-run)      DRY_RUN=1;;
    --yes|-y)       ASSUME_YES=1;;
    *) echo "unknown arg: $arg"; exit 1;;
  esac
done

: "${ASCEND_OPP_PATH:?ASCEND_OPP_PATH not set (source the CANN set_env.sh)}"
KROOT="${ASCEND_OPP_PATH}/built-in/op_impl/ai_core/tbe/kernel/${SOC}"
[[ -d "$KROOT" ]] || { echo "kernel root not found: $KROOT"; exit 1; }
[[ -f "$LIST" ]]  || { echo "list not found: $LIST"; exit 1; }

BK="${BACKUP_DIR}/disabled_kernels/${SOC}"
mkdir -p "$BK"
MANIFEST="${BK}/.manifest.txt"

echo "SOC=${SOC}"
echo "KROOT=${KROOT}"
echo "BACKUP=${BK}"
echo "DRY_RUN=${DRY_RUN}"
echo "------------------------------------------------------------"

# 风险确认：本脚本会修改系统 CANN 安装，绝不应在无人值守 / 共享机器上误触发。
if [[ $DRY_RUN -eq 0 ]]; then
  cat <<'WARN'
=============================== ⚠️  风险提示（务必阅读） ===============================
本操作会把系统 CANN 安装目录 (ASCEND_OPP_PATH) 下【被评测算子】的内置 kernel 二进制
用 mv 移动到 BACKUP 目录（即从 OPP 移除，不使用 rm，数据不会被删除）：
  * 修改的是【全局共享】的 CANN 安装，会影响本机所有进程 / 用户 / 其它项目；
  * 改动持续存在（不随进程退出自动恢复），仅能通过 restore_builtin_kernels.sh 还原；
  * 移除后，torch_npu 等在 NPU 上调用这些算子会直接报错（找不到 kernel）。
强烈建议：仅在【一次性 docker 容器 / 专用评测机】中执行，不要在共享开发机上直接运行。
本操作可逆：每个目录都是 mv 到 BACKUP 目录，可用 restore 脚本一键移回。
=====================================================================================
WARN
  if [[ $ASSUME_YES -eq 1 ]]; then
    echo "[--yes] 已确认，继续执行。"
  elif [[ -t 0 ]]; then
    read -r -p '确认删除以上内置 kernel？请输入大写 DELETE 以继续： ' _ans
    [[ "${_ans:-}" == "DELETE" ]] || { echo "未确认（输入非 DELETE），已取消。"; exit 1; }
  else
    echo "[ERROR] 非交互环境（无 TTY）且未指定 --yes：为防止误删已中止。" >&2
    echo "        如确需在脚本/容器中自动执行，请显式追加 --yes。" >&2
    exit 1
  fi
fi

removed=0; missing=0; already=0
while IFS= read -r rel; do
  rel="${rel%%#*}"; rel="$(echo "$rel" | xargs)"   # strip comments / whitespace
  [[ -z "$rel" ]] && continue
  src="${KROOT}/${rel}"
  dst="${BK}/${rel}"
  if [[ ! -e "$src" ]]; then
    if [[ -e "$dst" ]]; then already=$((already+1));   # already disabled (backup present)
    else echo "  MISSING (not present): ${rel}"; missing=$((missing+1)); fi
    continue
  fi
  if [[ $DRY_RUN -eq 1 ]]; then echo "  would disable: ${rel}"; removed=$((removed+1)); continue; fi
  mkdir -p "$(dirname "$dst")"
  if [[ -e "$dst" ]]; then
    # 备份已存在：保护原始备份，不覆盖、不动当前 src（避免任何数据丢失）。
    echo "  [WARN] 备份已存在，跳过(保护原始备份；如需重新禁用请先 restore): ${rel}"
    already=$((already+1))
    continue
  fi
  mv "$src" "$dst"      # 用 mv：把内置 kernel 移到备份目录 = 备份 + 从 OPP 移除（不使用 rm）
  echo "${rel}" >> "$MANIFEST"
  echo "  disabled (moved to backup): ${rel}"
  removed=$((removed+1))
done < "$LIST"

echo "------------------------------------------------------------"
echo "disabled=${removed}  already-disabled=${already}  missing=${missing}"
[[ $DRY_RUN -eq 0 ]] && echo "manifest: ${MANIFEST}"
echo "restore with: bash ${SCRIPT_DIR}/restore_builtin_kernels.sh --soc=${SOC} --backup-dir=${BACKUP_DIR}"
