#!/bin/bash
# S1 validation queue: C1 -> C2 -> C3 -> C5 (serial, one GPU) — final drivers.
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
echo "=== QUEUE C1 $(date -u +%H:%M:%S)"
ASYM_GEMM_DISPATCH=staged bash scripts/lf/tp_probe.sh q3-32b mrgs1c1 "asym_cpuadamwds|unsloth-ohbm0|ligerloss1" 128000 2
echo "C1_EXIT=$?"
echo "=== QUEUE C2 $(date -u +%H:%M:%S)"
ASYM_GEMM_DISPATCH=staged ASYMM_DENSE_MLP_FG_KEEP_ACTS_HBM=1 ASYM_SAVED_TENSOR_ASYNC_UNPACK=1 ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1 ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024 bash scripts/lf/tp_probe.sh q3-32b mrgs1c2 "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 128000 2
echo "C2_EXIT=$?"
echo "=== QUEUE C3 $(date -u +%H:%M:%S)"
ASYM_GEMM_DISPATCH=staged ASYMM_QWEN3_MOE_FG_KEEP_ACTS_HBM=1 ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024 ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1 bash scripts/lf/tp_probe.sh q3-30b-a3b mrgs1c3 "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 120000 8
echo "C3_EXIT=$?"
echo "=== QUEUE C5 $(date -u +%H:%M:%S)"
bash scripts/lf/tp_probe.sh q3-32b mrgs1c5 "superoffload_mem|unsloth-ohbm0|ligerloss1" 128000 2
echo "C5_EXIT=$?"
echo "=== S1 QUEUE DONE $(date -u +%H:%M:%S)"
