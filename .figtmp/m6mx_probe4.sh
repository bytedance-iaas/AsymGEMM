#!/bin/bash
# T3-outlives-T2B campaign, probe round 4 (INSIDE container): post-fix ladder.
# ROOT CAUSE (round 3): torch CachingHostAllocator pow2-rounds every pinned
# block; the [S,6144] bf16 GC boundary root crosses the 4-GiB bucket at
# S=349,525 -> billed 8 GiB each -> ~+205 GB host cliff (July's (320k,352k]
# blowup). Team fix exists since 2026-07-25: exact_pinned.py.
# INFRA FIX ON BOTH RUNGS (house rule: engineering fixes apply everywhere):
#   ASYM_EXACT_PINNED=1 (weight/opt homes; mixtral bank ~-70 GB pow2 tax)
#   ASYM_EXACT_PINNED_ROOTS=1 (GC boundary roots ride the exact RootPool)
# [1] m6mxt2bx384: T2B preset + infra fix @384K w1+m1 — the honest wall attempt
# [2] m6mxt3x384:  T3'' = T3 run-form + infra fix + tier dials
#     (token ohbm14, pool-cache 24 GB, guarded direct-reuse) @384K w1+m1
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM

MEMCSV=.figtmp/m6mx_hostmem4.csv
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

echo t2bx384 > /tmp/m6mx_phase
guard T2BX384
echo "=== [1] T2B+exactpinned mixtral 384K b1 capacity attempt (w1+m1) $(date -u +%H:%M:%S)"
ASYM_EXACT_PINNED=1 ASYM_EXACT_PINNED_ROOTS=1 \
WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 \
  bash scripts/lf/tp_probe.sh mixtral-8x22b m6mxt2bx384 "asym_cpuadamwds|T2B|ligerloss1" 384000 1
echo "M6MX4_T2BX_EXIT=$?"

echo t3x384 > /tmp/m6mx_phase
guard T3X384
echo "=== [2] T3'' (exactpinned+ohbm14+pool24+reuse) mixtral 384K b1 (w1+m1) $(date -u +%H:%M:%S)"
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
  bash scripts/lf/tp_probe.sh mixtral-8x22b m6mxt3x384 "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm14|ligerloss1" 384000 1
echo "M6MX4_T3X_EXIT=$?"
echo "=== PROBE ROUND 4 DONE $(date -u +%H:%M:%S)"
