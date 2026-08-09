#!/bin/bash
# T3' row consistency runs (INSIDE container) — fire ONLY after m6mxt3p384a FIT.
# The plotted T3 bars must be one config across the row: re-measure 320K and
# 192K under the exact T3' recipe (token ohbm14 + pool24 + direct-reuse + moe|T3
# env). w1+m2 timed protocol, MAX_SAMPLES=512, fresh tags.
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

T3P_ENV() {
  ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024 \
  ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1 \
  ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0 \
  ASYMM_QWEN3_MOE_FG_DA_GPU=1 \
  ASYMM_QWEN3_MOE_FG_KEEP_DGRADS_HBM=1 \
  ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU=1 \
  ASYM_CPU_OPS_THREADS=48 \
  ASYM_PLACEMENT_POLICY=1 \
  ASYM_EXPACT_CPU_POOL_MAX_BYTES=24000000000 \
  ASYMM_MOE_FG_DIRECT_REUSE=1 \
  MAX_SAMPLES=512 \
  "$@"
}

guard R320
echo "=== R1: T3' mixtral 320K b1 (w1+m2) $(date -u +%H:%M:%S)"
T3P_ENV bash scripts/lf/tp_probe.sh mixtral-8x22b m6mxt3p320 "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm14|ligerloss1" 320000 1
echo "M6MXR_320_EXIT=$?"

guard R192
echo "=== R2: T3' mixtral 192K b1 (w1+m2) $(date -u +%H:%M:%S)"
T3P_ENV bash scripts/lf/tp_probe.sh mixtral-8x22b m6mxt3p192 "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm14|ligerloss1" 192000 1
echo "M6MXR_192_EXIT=$?"
echo "=== T3' ROW CHAIN DONE $(date -u +%H:%M:%S)"
