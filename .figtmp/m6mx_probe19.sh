#!/bin/bash
# Round 17 (INSIDE container): slab-era corridor. Round-16 facts @416K:
# T2B dead at fwd-START (staged-dispatch host pools pre-size with seq);
# T3 died at bwd-layer ~11, short ~10-40 GB. Pair @384K, ascend 400 / descend 368.
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
guard() {
  sleep 30
  local apps
  for _ in $(seq 1 21); do
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
    [ -z "$apps" ] && return 0
    echo "!!! GUARD: GPU busy before $1 (pids: $apps) — waiting"; sleep 30
  done
  echo "!!! GUARD FAIL before $1 — aborting chain"; exit 9
}
XPC="ASYM_EXACT_PINNED=1 ASYM_EXACT_PINNED_ROOTS=1 ASYM_EXACT_PINNED_SAVED=1 ASYM_EXACT_PINNED_FORCE_CLONE=1"
t3run() { # $1 = seq
  guard "T3_$1"
  echo "=== T3(slab,ohbm56,blockexp,cap,skip) @$1 (w1+m1) $(date -u +%H:%M:%S)"
  env $XPC \
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
  WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 \
    bash scripts/lf/tp_probe.sh mixtral-8x22b "m6mxg5t3$(( $1 / 1000 ))" "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm56|ligerloss1" "$1" 1
}
t2brun() { # $1 = seq
  guard "T2B_$1"
  echo "=== T2B(slab) @$1 capacity attempt (w1+m1) $(date -u +%H:%M:%S)"
  env $XPC WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 \
    bash scripts/lf/tp_probe.sh mixtral-8x22b "m6mxg5t2b$(( $1 / 1000 ))" "asym_cpuadamwds|T2B|ligerloss1" "$1" 1
}
pair() { # 0 winner / 1 t3-dead / 2 both-fit
  if t3run "$1"; then
    if t2brun "$1"; then echo "PAIR@$1: BOTH FIT"; return 2
    else echo "PROBE19_WINNER=$1"; return 0; fi
  else echo "PAIR@$1: T3 dead"; return 1; fi
}
set +e
pair 356000; rc=$?
[ $rc -eq 0 ] && exit 0
if [ $rc -eq 2 ]; then
  pair 364000; rc=$?
  [ $rc -eq 0 ] && exit 0
  echo "PROBE19_STATE=rc$rc@364"; exit 3
fi
pair 352000; rc=$?
[ $rc -eq 0 ] && exit 0
echo "PROBE19_STATE=rc$rc@352"
exit 3
