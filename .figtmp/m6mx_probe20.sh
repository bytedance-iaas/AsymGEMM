#!/bin/bash
# Round 20 (INSIDE container): the 349K micro-corridor under best infra (slab).
# 349000*6144 = 2.144e9 < 2^31 — just under the pow2/copy cliff, where T3's
# dial stack (ohbm14 fits GPU here, skip, cap, blockexp, reuse) delivers in
# full while T2B carries its seq-scaled staged pools with no dials.
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
set +e
guard T3_349
echo "=== T3(slab,ohbm14,blockexp,cap,skip) @349K (w1+m1) $(date -u +%H:%M:%S)"
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
  bash scripts/lf/tp_probe.sh mixtral-8x22b m6mxg6t3349 "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm14|ligerloss1" 349000 1
T3E=$?
echo "R20_T3_349_EXIT=$T3E"
guard T2B_349
echo "=== T2B(slab) @349K capacity attempt (w1+m1) $(date -u +%H:%M:%S)"
env $XPC WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 \
  bash scripts/lf/tp_probe.sh mixtral-8x22b m6mxg6t2b349 "asym_cpuadamwds|T2B|ligerloss1" 349000 1
T2E=$?
echo "R20_T2B_349_EXIT=$T2E"
if [ $T3E -eq 0 ] && [ $T2E -ne 0 ]; then echo "PROBE20_WINNER=349000"; exit 0; fi
echo "PROBE20_STATE=t3=$T3E,t2b=$T2E"
exit 3
