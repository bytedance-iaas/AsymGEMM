#!/bin/bash
# FINAL mixtral row re-measure under uniform fix-2 infra (INSIDE container).
# Fire only after the round-14 separation is proven at ${L3K}K.
# Infra everywhere: ASYM_EXACT_PINNED=1 + _ROOTS=1 + _SAVED=1.
# T3 (plotted) recipe: token ohbm9 + block-experts scatter + cpu-act rows cap
# + skip-in-backward + direct-reuse + 64 worker threads. NO dual-DA (illegal
# access on mixtral, m6mxf8t3350).
# Usage: L3=350000 bash m6mx_row3.sh  (L3 = the sole-survivor length)
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
L3=${L3:-350000}
L3K=$(( L3 / 1000 ))

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

XP="ASYM_EXACT_PINNED=1 ASYM_EXACT_PINNED_ROOTS=1 ASYM_EXACT_PINNED_SAVED=1 ASYM_EXACT_PINNED_FORCE_CLONE=1"

T3ENV() {
  ASYM_EXACT_PINNED=1 ASYM_EXACT_PINNED_ROOTS=1 ASYM_EXACT_PINNED_SAVED=1 ASYM_EXACT_PINNED_FORCE_CLONE=1 \
  ASYM_ATTN_SAVED_TENSOR_OFFLOAD_SKIP_IN_BACKWARD=1 \
  ASYMM_QWEN3_MOE_FG_CPU_ACT_MAX_ROWS=500000 \
  ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024 \
  ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1 \
  ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=1 \
  ASYMM_QWEN3_MOE_FG_DA_GPU=1 \
  ASYMM_QWEN3_MOE_FG_KEEP_DGRADS_HBM=1 \
  ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU=1 \
  ASYM_CPU_OPS_THREADS=64 \
  ASYM_PLACEMENT_POLICY=1 \
  ASYMM_MOE_FG_DIRECT_REUSE=1 \
  MAX_SAMPLES=512 \
  "$@"
}

guard R1
echo "=== R1 T1 192K timed $(date -u +%H:%M:%S)"
env $XP MAX_SAMPLES=512 bash scripts/lf/tp_probe.sh mixtral-8x22b m6mxu2t1192 "asym_cpuadamwds|T1|ligerloss1" 192000 1
echo "ROW3_R1_EXIT=$?"

guard R2
echo "=== R2 T1 320K wall attempt (w1+m1) $(date -u +%H:%M:%S)"
env $XP WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 bash scripts/lf/tp_probe.sh mixtral-8x22b m6mxu2t1320 "asym_cpuadamwds|T1|ligerloss1" 320000 1
echo "ROW3_R2_EXIT=$?"

guard R3
echo "=== R3 T2B 192K timed $(date -u +%H:%M:%S)"
env $XP MAX_SAMPLES=512 bash scripts/lf/tp_probe.sh mixtral-8x22b m6mxu2t2b192 "asym_cpuadamwds|T2B|ligerloss1" 192000 1
echo "ROW3_R3_EXIT=$?"

guard R4
echo "=== R4 T2B 320K timed $(date -u +%H:%M:%S)"
env $XP MAX_SAMPLES=512 bash scripts/lf/tp_probe.sh mixtral-8x22b m6mxu2t2b320 "asym_cpuadamwds|T2B|ligerloss1" 320000 1
echo "ROW3_R4_EXIT=$?"

guard R5
echo "=== R5 T3 192K timed $(date -u +%H:%M:%S)"
T3ENV bash scripts/lf/tp_probe.sh mixtral-8x22b m6mxu2t3192 "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm56|ligerloss1" 192000 1
echo "ROW3_R5_EXIT=$?"

guard R6
echo "=== R6 T3 320K timed $(date -u +%H:%M:%S)"
T3ENV bash scripts/lf/tp_probe.sh mixtral-8x22b m6mxu2t3320 "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm56|ligerloss1" 320000 1
echo "ROW3_R6_EXIT=$?"

guard R7
echo "=== R7 T3 ${L3K}K timed (sole-survivor cell) $(date -u +%H:%M:%S)"
T3ENV bash scripts/lf/tp_probe.sh mixtral-8x22b "m6mxu2t3${L3K}" "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm56|ligerloss1" "$L3" 1
echo "ROW3_R7_EXIT=$?"
echo "=== ROW3 CHAIN DONE $(date -u +%H:%M:%S)"
