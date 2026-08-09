#!/bin/bash
# Mixtral T3 @192K FIXED (INSIDE container): the moe|T3 preset token ker101 is
# qwen-only (driver hard-rejects it for Mixtral; family T3 = generic engine,
# model_integration.md). Run the T3 recipe explicitly: ker000 token + the
# moe|T3 env from tier_recipes.sh (NO ASYM_GEMM_DISPATCH=staged -> weights
# streamed; the qwen3-named fg vars are inert on Mixtral, kept for fidelity).
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM

sleep 30
for _ in $(seq 1 21); do
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
  [ -z "$apps" ] && break
  echo "!!! GUARD: GPU busy (pids: $apps) — waiting"
  sleep 30
done

echo "=== T3fix: mixtral 192K b1 (w1+m2, ker000 + moe|T3 env) $(date -u +%H:%M:%S)"
ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024 \
ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1 \
ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0 \
ASYMM_QWEN3_MOE_FG_DA_GPU=1 \
ASYMM_QWEN3_MOE_FG_KEEP_DGRADS_HBM=1 \
ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU=1 \
ASYM_CPU_OPS_THREADS=48 \
ASYM_PLACEMENT_POLICY=1 \
MAX_SAMPLES=512 \
  bash scripts/lf/tp_probe.sh mixtral-8x22b m6mxt3192b "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 192000 1
echo "M6MX_T3FIX_EXIT=$?"
echo "=== T3FIX DONE $(date -u +%H:%M:%S)"
