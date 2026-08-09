#!/bin/bash
# T3-outlives-T2B campaign, probe round 1 (INSIDE container).
# P-A m6mxp1: T3 (mixtral run-form: ker000 token + moe|T3 env) with
#   ASYMM_FG_ELEMENTWISE_CHUNK_MB=256 (preset uses 1024) @320K w1+m1 —
#   calibrates the GPU trim of the ~70 GiB routed-expert workspace + frag,
#   and any host-side effect, against the banked m6mxt3320 baseline.
# P-B m6mxt2b384: T2B preset @384K w1+m1 capacity attempt — expected host
#   watchdog abort (becomes the middle red mark of the sole-survivor column).
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM

guard() {
  sleep 30
  local apps
  for _ in $(seq 1 21); do
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
    [ -z "$apps" ] && return 0
    echo "!!! GUARD: GPU busy before $1 (pids: $apps) — waiting"
    sleep 30
  done
  echo "!!! GUARD FAIL before $1 — aborting chain"
  exit 9
}

guard PA
echo "=== P-A: T3+chunk256 mixtral 320K b1 (w1+m1) $(date -u +%H:%M:%S)"
ASYMM_FG_ELEMENTWISE_CHUNK_MB=256 \
ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1 \
ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0 \
ASYMM_QWEN3_MOE_FG_DA_GPU=1 \
ASYMM_QWEN3_MOE_FG_KEEP_DGRADS_HBM=1 \
ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU=1 \
ASYM_CPU_OPS_THREADS=48 \
ASYM_PLACEMENT_POLICY=1 \
WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 \
  bash scripts/lf/tp_probe.sh mixtral-8x22b m6mxp1 "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 320000 1
echo "M6MXP_A_EXIT=$?"

guard PB
echo "=== P-B: T2B preset mixtral 384K b1 capacity attempt (w1+m1) $(date -u +%H:%M:%S)"
WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 \
  bash scripts/lf/tp_probe.sh mixtral-8x22b m6mxt2b384 "asym_cpuadamwds|T2B|ligerloss1" 384000 1
echo "M6MXP_B_EXIT=$?"
echo "=== PROBE ROUND 1 DONE $(date -u +%H:%M:%S)"
