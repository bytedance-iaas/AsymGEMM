#!/bin/bash
# T3-outlives-T2B campaign, probe round 5 (INSIDE container): post-fix wall
# ladder with conditional descent/ascent. Infra on BOTH rungs:
# ASYM_EXACT_PINNED=1 + ASYM_EXACT_PINNED_ROOTS=1 (roots exact; bank stays
# stock-pinned on both — loader-owned aliases, mrg38on guard). T3'' adds its
# tier dials (token ohbm14, pool24, direct-reuse). All w1+m1 capacity probes.
# Emits PROBE5_WINNER=<L> when a length has T3'' FIT + T2B ABORT.
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM

MEMCSV=.figtmp/m6mx_hostmem5.csv
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
  echo "t3x$1" > /tmp/m6mx_phase
  guard "T3x$1"
  echo "=== T3'' @$1 (w1+m1) $(date -u +%H:%M:%S)"
  ASYM_EXACT_PINNED=1 ASYM_EXACT_PINNED_ROOTS=1 \
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
  WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 \
    bash scripts/lf/tp_probe.sh mixtral-8x22b "m6mxt3x$(( $1 / 1000 ))" "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm14|ligerloss1" "$1" 1
}

t2brun() { # $1 = seq
  echo "t2bx$1" > /tmp/m6mx_phase
  guard "T2Bx$1"
  echo "=== T2B+fix @$1 capacity attempt (w1+m1) $(date -u +%H:%M:%S)"
  ASYM_EXACT_PINNED=1 ASYM_EXACT_PINNED_ROOTS=1 \
  WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 \
    bash scripts/lf/tp_probe.sh mixtral-8x22b "m6mxt2bx$(( $1 / 1000 ))" "asym_cpuadamwds|T2B|ligerloss1" "$1" 1
}

pair() { # $1 = seq ; returns 0 winner / 1 t3-dead (descend) / 2 both-fit (ascend)
  if t3run "$1"; then
    if t2brun "$1"; then
      echo "PAIR@$1: BOTH FIT -> ascend"
      return 2
    else
      echo "PROBE5_WINNER=$1"
      return 0
    fi
  else
    echo "PAIR@$1: T3'' dead -> descend"
    return 1
  fi
}

set +e
pair 364000; rc=$?
if [ $rc -eq 0 ]; then echo "=== PROBE5 DONE (winner 364K) $(date -u +%H:%M:%S)"; exit 0; fi
if [ $rc -eq 2 ]; then
  pair 372000; rc=$?
  [ $rc -eq 0 ] && { echo "=== PROBE5 DONE (winner 372K) $(date -u +%H:%M:%S)"; exit 0; }
  [ $rc -eq 1 ] && { echo "PROBE5_STATE=window_(364,372)_thin_t3wall"; exit 3; }
  pair 380000; rc=$?
  [ $rc -eq 0 ] && { echo "=== PROBE5 DONE (winner 380K) $(date -u +%H:%M:%S)"; exit 0; }
  echo "PROBE5_STATE=ascent_exhausted_rc$rc"; exit 3
fi
# descend
pair 356000; rc=$?
[ $rc -eq 0 ] && { echo "=== PROBE5 DONE (winner 356K) $(date -u +%H:%M:%S)"; exit 0; }
[ $rc -eq 2 ] && { echo "PROBE5_STATE=window_(356,364)_thin_t2bwall"; exit 3; }
pair 352000; rc=$?
[ $rc -eq 0 ] && { echo "=== PROBE5 DONE (winner 352K) $(date -u +%H:%M:%S)"; exit 0; }
echo "PROBE5_STATE=descent_exhausted_rc$rc"
exit 3
