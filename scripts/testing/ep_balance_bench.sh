#!/bin/bash
# Driver for scripts/testing/ep_balance_bench.py (S5a balancing microbench):
# owned-static vs ownerless-queue grouped GEMM on REAL recorded routing.
#
# Usage (all knobs are env overrides; defaults reproduce the banked S5a runs):
#   bash scripts/testing/ep_balance_bench.sh
#   HIST=... ALPHAS=natural,0.10,0.15 GPUS=2,3 bash scripts/testing/ep_balance_bench.sh
#
# Knobs:
#   HIST    routing histogram json (capture one via ASYM_EP_STATS=1 on a |1 row)
#   LAYERS  worst,median[,best|<layer-key>]         (default worst,median)
#   ALPHAS  natural,0.05,...  injected hot-expert fractions (default natural,0.10,0.15,0.5,0.75)
#   MTOTAL  rows per case (default 5120000 — event-timed, floor-safe; do NOT go
#           below ~4M: the 1.28M first attempt was launch-floor-dominated)
#   REPS    timed reps per mode (default 3; rep0 JIT warm is always dropped)
#   GPUS    two comma GPUs (default 2,3 — keeps the training pair 0,1 free)
#   OUT     output json (default profiling_both_skew/ep_balance_bench_<ts>.json)
set -euo pipefail
cd "$(dirname "$0")/../.."

HIST="${HIST:-profiling_both_epstats/ep_hist_q3_s20000.json}"
LAYERS="${LAYERS:-worst,median}"
ALPHAS="${ALPHAS:-natural,0.10,0.15,0.5,0.75}"
MTOTAL="${MTOTAL:-5120000}"
REPS="${REPS:-3}"
GPUS="${GPUS:-2,3}"
OUT="${OUT:-profiling_both_skew/ep_balance_bench_$(date +%Y%m%d_%H%M%S).json}"

[[ -f "${HIST}" ]] || { echo "HIST not found: ${HIST} (capture: ASYM_EP_STATS=1 ASYM_EP_STATS_PATH=<path> on a |1 row)" >&2; exit 2; }
[[ "${GPUS}" == *,* ]] || { echo "GPUS must name two GPUs, e.g. 2,3 (got '${GPUS}')" >&2; exit 2; }

echo "[ep_balance_bench] hist=${HIST} layers=${LAYERS} alphas=${ALPHAS} m=${MTOTAL} reps=${REPS} gpus=${GPUS}"
mkdir -p "$(dirname "${OUT}")"
echo "[ep_balance_bench] out=${OUT}"

exec .venv/bin/python scripts/testing/ep_balance_bench.py \
  --hist "${HIST}" \
  --layers "${LAYERS}" \
  --alphas "${ALPHAS}" \
  --m-total "${MTOTAL}" \
  --reps "${REPS}" \
  --gpus "${GPUS}" \
  --out "${OUT}"
