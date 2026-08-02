#!/usr/bin/env bash
# Serial capacity-push chain (2026-07-25). Runs cells one at a time with the
# full fix set ON, stopping a phase's ladder on its first non-OK (climbing a
# ladder past a death wastes hours). Usage: run_push_chain.sh PHASE
#   PHASE=V23  -> V2 (exact-pinned) then V3 (+auto-park) at 97.891B/128k
#   PHASE=X    -> 128k ladder: X1 217.948 / X2 233.249 / X3 248.550 / X4 263.851 / X5 279.152
#   PHASE=Y    -> 64k ladder:  233.249 / 263.851 / 294.453 / 309.754 / 325.055 / 340.356 / 355.657
# Reserve: env RESERVE_GB (45 @h<=9216, 60 @12288 — see model_capacity.md trap #16).
set -u
PHASE="$1"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
M=/scratch_local/user_data/shutian/kevin/models_synth
SPEC='asym_cpuadamwds|unsloth-ohbm0|ligerloss1'
export CAPDIR="${CAPDIR:-${ROOT}/profiling_results/capacity_push_c17}"
export ASYM_GEMM_DISPATCH=staged ASYM_HOST_FLUSH_EVERY_N_LAYERS=8 ASYM_CPU_ADAMW_ASYNC_GRAD_OFFLOAD=0
export ASYM_EXACT_PINNED=1 ASYM_EXACT_PINNED_ROOTS=1

run_cell() { # CELL MODEL SEQ [auto: 0 disables parking]
  local cell="$1" model="$2" seq="$3" auto="${4:-1}"
  export UNSLOTH_GC_OUTER_HBM_AUTO="${auto}" UNSLOTH_GC_OUTER_HBM_RESERVE_GB="${RESERVE_GB:-60}"
  TIMEOUT_S="${TIMEOUT_S:-4500}" bash "${ROOT}/scripts/lf/capacity/run_capacity_cell.sh" \
    "${cell}" "${M}/${model}" "${SPEC}" "${seq}" 1 25
  local verdict
  verdict=$(tail -1 "${CAPDIR}/ledger.tsv" | cut -f2)
  echo "CHAIN ${cell} -> ${verdict}"
  case "${verdict}" in C_OOM*|G_OOM|CRASH|UNKNOWN) return 1;; esac
  return 0
}

case "${PHASE}" in
  V23)
    run_cell V2_exact_98b_128k d2-dense-100b-88x9216 128000 0
    run_cell V3_exactpark_98b_128k d2-dense-100b-88x9216 128000 1
    ;;
  X)
    run_cell X1_218b_128k d5e-dense-215b-112x12288 128000 || exit 1
    run_cell X2_233b_128k d6a-dense-233b-120x12288 128000 || exit 1
    run_cell X3_249b_128k d6b-dense-249b-128x12288 128000 || exit 1
    run_cell X4_264b_128k d6c-dense-264b-136x12288 128000 || exit 1
    run_cell X5_279b_128k d6d-dense-279b-144x12288 128000 || exit 1
    ;;
  Y)
    run_cell Y1_233b_64k d6a-dense-233b-120x12288 64000 || exit 1
    run_cell Y2_264b_64k d6c-dense-264b-136x12288 64000 || exit 1
    run_cell Y3_295b_64k d6e-dense-295b-152x12288 64000 || exit 1
    run_cell Y4_310b_64k d6f-dense-310b-160x12288 64000 || exit 1
    run_cell Y5_325b_64k d6g-dense-325b-168x12288 64000 || exit 1
    run_cell Y6_340b_64k d6h-dense-340b-176x12288 64000 || exit 1
    run_cell Y7_356b_64k d6i-dense-356b-184x12288 64000 || exit 1
    ;;
  *) echo "unknown phase ${PHASE}"; exit 2;;
esac
echo "CHAIN ${PHASE} COMPLETE"
