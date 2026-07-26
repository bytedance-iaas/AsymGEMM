#!/bin/bash
# CPU-merge queue 1: S3 flags-off invariance (R2 + R8, policy UNSET)
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM

echo "=== S3a R2 flags-off (q32 T2 128k b2; baseline 986 tok/s / 93.6 GiB) $(date -u +%H:%M:%S)"
ASYM_GEMM_DISPATCH=staged ASYMM_DENSE_MLP_FG_KEEP_ACTS_HBM=1 ASYM_SAVED_TENSOR_ASYNC_UNPACK=1 ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1 ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024 bash scripts/lf/tp_probe.sh q3-32b cpum3a "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 128000 2
echo "S3A_EXIT=$?"

echo "=== S3b R8 flags-off (moe shed 800k b1; baseline 584 tok/s / 110.4 GiB) $(date -u +%H:%M:%S)"
ASYM_GEMM_DISPATCH=staged ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU=1 ASYMM_QWEN3_MOE_FG_DA_GPU=1 ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0 ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024 ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1 ASYMM_QWEN3_MOE_FG_KEEP_DGRADS_HBM=1 MAX_SAMPLES=1024 bash scripts/lf/tp_probe.sh q3-30b-a3b cpum3b "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 800000 1
echo "S3B_EXIT=$?"
echo "=== CPU-Q1 DONE $(date -u +%H:%M:%S)"
