#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------
# Undo disable_builtin_kernels.sh: restore every backed-up built-in kernel binary dir.
#
# Usage:
#   bash scripts/anti_cheat/restore_builtin_kernels.sh [--soc=ascend910b] [--backup-dir=<dir>] [--dry-run]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOC="ascend910b"
BACKUP_DIR="${CANN_BENCH_KERNEL_BACKUP:-$HOME/.cann_bench_kernel_backup}"
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --soc=*)        SOC="${arg#*=}";;
    --backup-dir=*) BACKUP_DIR="${arg#*=}";;
    --dry-run)      DRY_RUN=1;;
    *) echo "unknown arg: $arg"; exit 1;;
  esac
done

: "${ASCEND_OPP_PATH:?ASCEND_OPP_PATH not set}"
KROOT="${ASCEND_OPP_PATH}/built-in/op_impl/ai_core/tbe/kernel/${SOC}"
BK="${BACKUP_DIR}/disabled_kernels/${SOC}"
MANIFEST="${BK}/.manifest.txt"
[[ -f "$MANIFEST" ]] || { echo "no manifest at ${MANIFEST}; nothing to restore"; exit 0; }

echo "restoring from ${BK} -> ${KROOT} (dry_run=${DRY_RUN})"
echo "------------------------------------------------------------"
restored=0
while IFS= read -r rel; do
  [[ -z "$rel" ]] && continue
  src="${BK}/${rel}"; dst="${KROOT}/${rel}"
  [[ -e "$src" ]] || { echo "  backup missing: ${rel}"; continue; }
  if [[ $DRY_RUN -eq 1 ]]; then echo "  would restore: ${rel}"; restored=$((restored+1)); continue; fi
  mkdir -p "$(dirname "$dst")"
  if [[ -e "$dst" ]]; then
    # 原位置已存在(可能 CANN 已重装/已恢复)，跳过以免覆盖现有内容（不使用 rm）。
    echo "  [WARN] 原位置已存在，跳过以免覆盖: ${rel}"
    continue
  fi
  mv "$src" "$dst"      # 用 mv：把备份移回原位（不使用 rm）
  echo "  restored (moved back): ${rel}"
  restored=$((restored+1))
done < "$MANIFEST"
echo "------------------------------------------------------------"
echo "restored=${restored}"
[[ $DRY_RUN -eq 0 ]] && { : > "${MANIFEST}"; echo "manifest cleared"; }
