#!/bin/bash
# Round 13 (INSIDE container): the 350K pair. 350000 > 349,525 (the 4-GiB pow2
# line) so T2B's residual allocator-billed classes still cliff it (as @352K),
# while T3''''+block-experts carries ~25-40 GiB of dial relief with only ~2 GiB
# more linear demand than its 348K fit.
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
set +e
guard T3_350
echo "=== T3c(ohbm9,blockexp) @350K (w1+m1) $(date -u +%H:%M:%S)"
ASYM_EXACT_PINNED=1 ASYM_EXACT_PINNED_ROOTS=1 ASYM_EXACT_PINNED_SAVED=1 \
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
  bash scripts/lf/tp_probe.sh mixtral-8x22b m6mxf9t3350 "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm9|ligerloss1" 350000 1
T3EXIT=$?
echo "R14_T3_EXIT=$T3EXIT"
if [ $T3EXIT -ne 0 ]; then echo "PROBE14_STATE=t3_dead_350"; exit 3; fi
guard T2B_350
echo "=== T2B(fix2) @350K capacity attempt (w1+m1) $(date -u +%H:%M:%S)"
ASYM_EXACT_PINNED=1 ASYM_EXACT_PINNED_ROOTS=1 ASYM_EXACT_PINNED_SAVED=1 \
WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 \
  bash scripts/lf/tp_probe.sh mixtral-8x22b m6mxf9t2b350 "asym_cpuadamwds|T2B|ligerloss1" 350000 1
T2BEXIT=$?
echo "R14_T2B_EXIT=$T2BEXIT"
if [ $T2BEXIT -ne 0 ]; then echo "PROBE14_WINNER=350000"; exit 0; fi
echo "PROBE14_STATE=both_fit_350"
exit 3
