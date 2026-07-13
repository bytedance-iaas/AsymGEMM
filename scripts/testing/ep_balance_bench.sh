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
MTOTAL="${MTOTAL:-}"
REPS="${REPS:-3}"
SEEDS="${SEEDS:-3}"
MODES="${MODES:-owned,sdp,plan,queue}"   # "sep" accepted as a legacy alias of "plan"
SCOPE="${SCOPE:-gemm}"   # gemm | experts | moe | layer (attention + MoE block)
GPUS="${GPUS:-2,3}"
OUT="${OUT:-profiling_both_skew/ep_balance_bench_$(date +%Y%m%d_%H%M%S).json}"

# MODEL presets: MoE geometry VERIFIED from the HF configs (2026-07-10).
# GEOM=E,N,K; TOPK; FUSED (llama4 single gate_up GEMM); SHARED_N (shared-expert
# width, moe scope only). MTOTAL scales with row width to fit HBM and stay above
# the ~launch floor. Real-routing (natural) columns need a capture from the SAME
# model — pure-z sweeps skip HIST entirely.
MODEL="${MODEL:-q3-30b-a3b}"
case "${MODEL}" in
  q3-30b-a3b)   GEOM="128,768,2048";  TOPK=8; FUSED=0; SHARED_N=0;    MTOTAL="${MTOTAL:-5120000}" ;;
  q3-235b-a22b) GEOM="128,1536,4096"; TOPK=8; FUSED=0; SHARED_N=0;    MTOTAL="${MTOTAL:-3840000}" ;;
  q35-122b-a10b) GEOM="256,1024,3072"; TOPK=8; FUSED=0; SHARED_N=1024; MTOTAL="${MTOTAL:-5120000}" ;;
  l4-scout)     GEOM="16,8192,5120";  TOPK=1; FUSED=1; SHARED_N=8192; MTOTAL="${MTOTAL:-1280000}" ;;
  *) echo "unknown MODEL '${MODEL}' (q3-30b-a3b|q3-235b-a22b|q35-122b-a10b|l4-scout)" >&2; exit 2 ;;
esac

if [[ "${ALPHAS}" =~ (natural|(^|,)g|(^|,)0?\.) ]]; then
  [[ -f "${HIST}" ]] || { echo "HIST not found: ${HIST} (needed for natural/gamma/alpha columns; capture: ASYM_EP_STATS=1 on a |1 row of the SAME model)" >&2; exit 2; }
fi
[[ "${GPUS}" == *,* ]] || { echo "GPUS must name two GPUs, e.g. 2,3 (got '${GPUS}')" >&2; exit 2; }

echo "[ep_balance_bench] model=${MODEL} geom=${GEOM} topk=${TOPK} fused=${FUSED} shared_n=${SHARED_N}"
echo "[ep_balance_bench] hist=${HIST} layers=${LAYERS} alphas=${ALPHAS} modes=${MODES} scope=${SCOPE} m=${MTOTAL} reps=${REPS} gpus=${GPUS}"
mkdir -p "$(dirname "${OUT}")"
echo "[ep_balance_bench] out=${OUT}"

exec .venv/bin/python scripts/testing/ep_balance_bench.py \
  --hist "${HIST}" \
  --layers "${LAYERS}" \
  --alphas "${ALPHAS}" \
  --m-total "${MTOTAL}" \
  --reps "${REPS}" \
  --seeds "${SEEDS}" \
  --modes "${MODES}" \
  --scope "${SCOPE}" \
  --geom "${GEOM}" \
  --topk "${TOPK}" \
  --fused-gateup "${FUSED}" \
  --shared-n "${SHARED_N}" \
  --gpus "${GPUS}" \
  --out "${OUT}"
