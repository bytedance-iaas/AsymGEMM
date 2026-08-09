#!/bin/bash
# T3-outlives-T2B campaign, probe round 8 (INSIDE container).
# New T3'' dial: ASYM_ATTN_SAVED_TENSOR_OFFLOAD_SKIP_IN_BACKWARD=1 — the
# attention wrapper's in-backward offloads are documented duplicates
# (fix_qwen3.5 §10b C3); skipping keeps them GPU-side for their one-layer
# window (~11 GB @352K) and removes the pow2-taxed per-save host churn that
# survived fix-2 (deaths invariant at bwd-layer ~5, RSS ~978).
# Ladder: pair@352 -> ascend 360 -> 364. T2B stays the fix-2 preset (the skip
# is a tier dial, not infra).
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM

MEMCSV=.figtmp/m6mx_hostmem8.csv
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

t3run() { # $1 = seq
  echo "f3t3_$1" > /tmp/m6mx_phase
  guard "F3T3_$1"
  echo "=== T3''(fix2+skipbwd,ohbm14) @$1 (w1+m1) $(date -u +%H:%M:%S)"
  ASYM_EXACT_PINNED=1 ASYM_EXACT_PINNED_ROOTS=1 ASYM_EXACT_PINNED_SAVED=1 \
  ASYM_ATTN_SAVED_TENSOR_OFFLOAD_SKIP_IN_BACKWARD=1 \
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
    bash scripts/lf/tp_probe.sh mixtral-8x22b "m6mxf3t3$(( $1 / 1000 ))" "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm14|ligerloss1" "$1" 1
}

t2brun() { # $1 = seq
  echo "f3t2b_$1" > /tmp/m6mx_phase
  guard "F3T2B_$1"
  echo "=== T2B(fix2) @$1 capacity attempt (w1+m1) $(date -u +%H:%M:%S)"
  ASYM_EXACT_PINNED=1 ASYM_EXACT_PINNED_ROOTS=1 ASYM_EXACT_PINNED_SAVED=1 \
  WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 \
    bash scripts/lf/tp_probe.sh mixtral-8x22b "m6mxf3t2b$(( $1 / 1000 ))" "asym_cpuadamwds|T2B|ligerloss1" "$1" 1
}

pair() { # $1 = seq ; 0 winner / 1 t3-dead / 2 both-fit
  if t3run "$1"; then
    if t2brun "$1"; then echo "PAIR@$1: BOTH FIT"; return 2
    else echo "PROBE8_WINNER=$1"; return 0; fi
  else echo "PAIR@$1: T3'' dead"; return 1; fi
}

set +e
pair 352000; rc=$?
[ $rc -eq 0 ] && exit 0
[ $rc -eq 1 ] && { echo "PROBE8_STATE=t3_dead_352_skipbwd"; exit 3; }
pair 360000; rc=$?
[ $rc -eq 0 ] && exit 0
[ $rc -eq 1 ] && { echo "PROBE8_STATE=window_(352,360)"; exit 3; }
pair 364000; rc=$?
[ $rc -eq 0 ] && exit 0
echo "PROBE8_STATE=exhausted_rc$rc"
exit 3
