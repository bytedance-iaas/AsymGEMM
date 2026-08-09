#!/bin/bash
# Round 9 (INSIDE container): T3''@352K with the built-in 3-s host-attribution
# logger (exact_pinned ASYM_MEM_ATTRIB_LOG) + worker threads 48->64. One run,
# two answers: if it FITS, worker lag was the binding factor; if it dies, the
# attribution log names the accumulating class.
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
sleep 30
for _ in $(seq 1 21); do
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
  [ -z "$apps" ] && break
  echo "!!! GUARD: GPU busy (pids: $apps) — waiting"; sleep 30
done
echo "=== R9: T3''(fix2+skipbwd,ohbm14,threads64,attrib) @352K (w1+m1) $(date -u +%H:%M:%S)"
ASYM_EXACT_PINNED=1 ASYM_EXACT_PINNED_ROOTS=1 ASYM_EXACT_PINNED_SAVED=1 \
ASYM_ATTN_SAVED_TENSOR_OFFLOAD_SKIP_IN_BACKWARD=1 \
ASYM_MEM_ATTRIB_LOG=/workspace/AsymGEMM-SFT/third_party/AsymGEMM/.figtmp/attrib352.log \
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
  bash scripts/lf/tp_probe.sh mixtral-8x22b m6mxf4t3352 "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm14|ligerloss1" 352000 1
echo "M6MX9_EXIT=$?"
echo "=== ROUND 9 DONE $(date -u +%H:%M:%S)"
