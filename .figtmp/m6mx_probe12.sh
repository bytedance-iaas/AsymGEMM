#!/bin/bash
# Round 12 (INSIDE container): the decisive 352K corridor.
# Facts: T3''-fix2@352 died by ~1 GiB (watchdog 49 vs floor 50); pre-fix
# T2B@348 FIT; the fg gate/up host saves are the engine's unconditional save
# design (~48 GiB/layer at 352K, 1-2 layer window). Final levers stacked on
# T3: ASYMM_ATTN_DUAL_DA=1 (rank-64 confirmed; drops ~10 GiB pinned S bufs)
# + ohbm9 (~6 roots in HBM, -9-13 GiB host, GPU ~179 GiB in-band @352K).
# [1] T2B-fix2@352K — the linchpin wall attempt (expect HOST abort)
# [2] T3''''@352K   — expect FIT
# [3] if both as expected -> WINNER=352K; if T2B fits -> its wall via 360K...
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

t2brun() { # $1 = seq
  guard "T2B_$1"
  echo "=== T2B(fix2) @$1 capacity attempt (w1+m1) $(date -u +%H:%M:%S)"
  ASYM_EXACT_PINNED=1 ASYM_EXACT_PINNED_ROOTS=1 ASYM_EXACT_PINNED_SAVED=1 \
  WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 \
    bash scripts/lf/tp_probe.sh mixtral-8x22b "m6mxf7t2b$(( $1 / 1000 ))" "asym_cpuadamwds|T2B|ligerloss1" "$1" 1
}

t3run() { # $1 = seq
  guard "T3_$1"
  echo "=== T3''''(dualda,ohbm9) @$1 (w1+m1) $(date -u +%H:%M:%S)"
  ASYM_EXACT_PINNED=1 ASYM_EXACT_PINNED_ROOTS=1 ASYM_EXACT_PINNED_SAVED=1 \
  ASYM_ATTN_SAVED_TENSOR_OFFLOAD_SKIP_IN_BACKWARD=1 \
  ASYMM_ATTN_DUAL_DA=1 \
  ASYMM_QWEN3_MOE_FG_CPU_ACT_MAX_ROWS=500000 \
  ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024 \
  ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1 \
  ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0 \
  ASYMM_QWEN3_MOE_FG_DA_GPU=1 \
  ASYMM_QWEN3_MOE_FG_KEEP_DGRADS_HBM=1 \
  ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU=1 \
  ASYM_CPU_OPS_THREADS=64 \
  ASYM_PLACEMENT_POLICY=1 \
  ASYMM_MOE_FG_DIRECT_REUSE=1 \
  WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 \
    bash scripts/lf/tp_probe.sh mixtral-8x22b "m6mxf7t3$(( $1 / 1000 ))" "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm9|ligerloss1" "$1" 1
}

set +e
if t2brun 352000; then
  echo "T2B@352 FIT -> find its wall at 360"
  if t2brun 360000; then
    echo "PROBE12_STATE=t2b_fits_360_too"; exit 3
  else
    if t3run 360000; then echo "PROBE12_WINNER=360000"; exit 0
    else echo "PROBE12_STATE=t2b_dead_360_t3_dead_360"; exit 3; fi
  fi
else
  echo "T2B@352 DEAD (wall attempt banked)"
  if t3run 352000; then
    echo "PROBE12_WINNER=352000"; exit 0
  else
    echo "PROBE12_STATE=both_dead_352"; exit 3
  fi
fi
