#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <kernel> <report-path> [profile_exp_act_offload_kernel_reuse.py args...]" >&2
  exit 2
fi

kernel="$1"
report="$2"
shift 2

ncu \
  --target-processes all \
  --set full \
  --launch-skip 1 \
  --launch-count 1 \
  --force-overwrite \
  --export "${report}" \
  python scripts/testing/profile_exp_act_offload_kernel_reuse.py \
    --kernel "${kernel}" \
    --iters 3 \
    "$@"
