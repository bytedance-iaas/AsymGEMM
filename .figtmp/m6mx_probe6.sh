#!/bin/bash
# T3-outlives-T2B campaign, probe round 6 (INSIDE container): fix-2 ladder.
# Infra on BOTH rungs: ASYM_EXACT_PINNED=1 + _ROOTS=1 + _SAVED=1
# (_SAVED is new: activation-offload pool buffers exact-registered — kills the
# >4-GiB pow2 tax on q/o_proj.U and moe.X/act/gate/up past S=349,525; unit-
# validated pintest2: 4.66 GB billed vs 8, single registration, trim-immune).
# T3'' tier dials: token ohbm14 + ASYMM_MOE_FG_DIRECT_REUSE=1 (pool cap dropped
# — registered buffers never evict). All w1+m1. Conditional:
#   pair@384K: T3'' fit + T2B abort -> WINNER 384K
#   both fit -> pair@400K (T3'' falls back to ohbm28 there: GPU envelope)
#   T3'' dead -> pair@368K
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM

MEMCSV=.figtmp/m6mx_hostmem6.csv
echo "epoch,phase,node0_memfree_kb,global_memavailable_kb,top_rss_kb" >> "$MEMCSV"
echo init > /tmp/m6mx_phase
( while :; do
    nf=$(awk '/MemFree/{print $4}' /sys/devices/system/node/node0/meminfo 2>/dev/null)
    ga=$(awk '/MemAvailable/{print $2}' /proc/meminfo 2>/dev/null)
    tr=$(ps -eo rss,comm --sort=-rss 2>/dev/null | awk '/python/{print $1; exit}')
    echo "$(date +%s),$(cat /tmp/m6mx_phase 2>/dev/null || echo '?'),${nf:-0},${ga:-0},${tr:-0}" >> "$MEMCSV"
    sleep 2
  done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

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

t3run() { # $1 = seq, $2 = ohbm N
  echo "f2t3_$1" > /tmp/m6mx_phase
  guard "F2T3_$1"
  echo "=== T3''(fix2,ohbm$2) @$1 (w1+m1) $(date -u +%H:%M:%S)"
  ASYM_EXACT_PINNED=1 ASYM_EXACT_PINNED_ROOTS=1 ASYM_EXACT_PINNED_SAVED=1 \
  ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024 \
  ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1 \
  ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0 \
  ASYMM_QWEN3_MOE_FG_DA_GPU=1 \
  ASYMM_QWEN3_MOE_FG_KEEP_DGRADS_HBM=1 \
  ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU=1 \
  ASYM_CPU_OPS_THREADS=48 \
  ASYM_PLACEMENT_POLICY=1 \
  ASYMM_MOE_FG_DIRECT_REUSE=1 \
  WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 \
    bash scripts/lf/tp_probe.sh mixtral-8x22b "m6mxf2t3$(( $1 / 1000 ))" "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm$2|ligerloss1" "$1" 1
}

t2brun() { # $1 = seq
  echo "f2t2b_$1" > /tmp/m6mx_phase
  guard "F2T2B_$1"
  echo "=== T2B(fix2) @$1 capacity attempt (w1+m1) $(date -u +%H:%M:%S)"
  ASYM_EXACT_PINNED=1 ASYM_EXACT_PINNED_ROOTS=1 ASYM_EXACT_PINNED_SAVED=1 \
  WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 \
    bash scripts/lf/tp_probe.sh mixtral-8x22b "m6mxf2t2b$(( $1 / 1000 ))" "asym_cpuadamwds|T2B|ligerloss1" "$1" 1
}

set +e
if t3run 384000 14; then
  if t2brun 384000; then
    echo "PAIR@384: BOTH FIT -> ascend 400K"
    if t3run 400000 28; then
      if t2brun 400000; then echo "PROBE6_STATE=both_fit_400"; exit 3
      else echo "PROBE6_WINNER=400000"; exit 0; fi
    else echo "PROBE6_STATE=window_(384,400)_t3wall"; exit 3; fi
  else
    echo "PROBE6_WINNER=384000"; exit 0
  fi
else
  echo "PAIR@384: T3'' dead -> descend 368K"
  if t3run 368000 14; then
    if t2brun 368000; then echo "PROBE6_STATE=both_fit_368_window_thin"; exit 3
    else echo "PROBE6_WINNER=368000"; exit 0; fi
  else
    echo "PROBE6_STATE=t3_dead_368_too"; exit 3
  fi
fi
