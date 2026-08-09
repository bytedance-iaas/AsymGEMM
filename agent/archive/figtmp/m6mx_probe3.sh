#!/bin/bash
# T3-outlives-T2B campaign, probe round 3 (INSIDE container): cliff bracketing.
# The [B*S, H] boundary flat crosses 2^31 elements at S=349,525 (H=6144).
# July's host cliff window was (320k, 352k]. Discriminating triplet:
#   [1] m6mxt3p352a: T3' @352K w1+m1 — just ABOVE 2^31 → dies if theory right
#   [2] m6mxt3p348a: T3' @348K w1+m1 — just BELOW 2^31 (same 6x65536 row
#       bucket as 352K/384K, so a bucket-cliff would kill this too) → fits
#       if 2^31 theory right
#   [3] m6mxt2b348:  T2B preset @348K w1+m1 — the separation test at 348K
# A host-memory sampler (2 s cadence) records node0 free + fattest-python RSS
# to .figtmp/m6mx_hostmem.csv for exact standing/trough curves.
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM

MEMCSV=.figtmp/m6mx_hostmem.csv
echo "epoch,phase,node0_memfree_kb,global_memavailable_kb,top_py_pid,top_py_vmrss_kb" >> "$MEMCSV"
PHASE=init
( while :; do
    nf=$(awk '/MemFree/{print $4}' /sys/devices/system/node/node0/meminfo 2>/dev/null)
    ga=$(awk '/MemAvailable/{print $2}' /proc/meminfo 2>/dev/null)
    read -r pid rss <<< "$(for p in /proc/[0-9]*; do c=$(tr -d '\0' < $p/cmdline 2>/dev/null | head -c 60); case "$c" in *python*) r=$(awk '/VmRSS/{print $2}' $p/status 2>/dev/null); echo "${r:-0} ${p#/proc/}";; esac; done | sort -rn | head -1 | awk '{print $2, $1}')"
    echo "$(date +%s),$(cat /tmp/m6mx_phase 2>/dev/null || echo '?'),${nf:-0},${ga:-0},${pid:-0},${rss:-0}" >> "$MEMCSV"
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
  WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 \
  "$@"
}

echo p352 > /tmp/m6mx_phase
guard P352
echo "=== [1] T3' mixtral 352K b1 (w1+m1, ohbm14+pool24+reuse) $(date -u +%H:%M:%S)"
T3P_ENV bash scripts/lf/tp_probe.sh mixtral-8x22b m6mxt3p352a "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm14|ligerloss1" 352000 1
echo "M6MX3_352_EXIT=$?"

echo p348 > /tmp/m6mx_phase
guard P348
echo "=== [2] T3' mixtral 348K b1 (w1+m1, ohbm14+pool24+reuse) $(date -u +%H:%M:%S)"
T3P_ENV bash scripts/lf/tp_probe.sh mixtral-8x22b m6mxt3p348a "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm14|ligerloss1" 348000 1
echo "M6MX3_348_EXIT=$?"

echo t2b348 > /tmp/m6mx_phase
guard T2B348
echo "=== [3] T2B preset mixtral 348K b1 capacity attempt (w1+m1) $(date -u +%H:%M:%S)"
WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 \
  bash scripts/lf/tp_probe.sh mixtral-8x22b m6mxt2b348 "asym_cpuadamwds|T2B|ligerloss1" 348000 1
echo "M6MX3_T2B348_EXIT=$?"
echo "=== PROBE ROUND 3 DONE $(date -u +%H:%M:%S)"
