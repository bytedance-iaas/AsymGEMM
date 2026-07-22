#!/bin/bash
# CPU-merge queue 3: S6 — the 9-row no-regression matrix (M1-M9).
# Each row = previous_validation_results.md config VERBATIM + the CPU stack
# (ASYM_PLACEMENT_POLICY=1 ASYM_CPU_OPS_THREADS=48), w1+m2 protocol.
# Order: fast dense rows first, deep rows last (fail-fast on cheap rows).
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
POL="ASYM_PLACEMENT_POLICY=1 ASYM_CPU_OPS_THREADS=48"
T2D="ASYM_GEMM_DISPATCH=staged ASYMM_DENSE_MLP_FG_KEEP_ACTS_HBM=1 ASYM_SAVED_TENSOR_ASYNC_UNPACK=1 ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1 ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024"
PINS="ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU=1 ASYMM_QWEN3_MOE_FG_DA_GPU=1 ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0 ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024 ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1 ASYMM_QWEN3_MOE_FG_KEEP_DGRADS_HBM=1"
# M7 uses the R7 dial env VERBATIM (mrgs1c3): KA(FG)+chunk+dx only — NOT the T2 bundle

echo "=== M1 q32 T1 128k b2 (base 1091 / 116.0) $(date -u +%H:%M:%S)"
env $POL ASYM_GEMM_DISPATCH=staged bash scripts/lf/tp_probe.sh q3-32b cpum1 "asym_cpuadamwds|unsloth-ohbm0|ligerloss1" 128000 2
echo "M1_EXIT=$?"

echo "=== M2 q32 T2 128k b2 (base 986 / 93.6) $(date -u +%H:%M:%S)"
env $POL $T2D bash scripts/lf/tp_probe.sh q3-32b cpum2 "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 128000 2
echo "M2_EXIT=$?"

echo "=== M4 llama T1 96k b1 (base 1096 / 48.9) $(date -u +%H:%M:%S)"
env $POL ASYM_GEMM_DISPATCH=staged MAX_SAMPLES=512 bash scripts/lf/tp_probe.sh llama3.3-70b cpum4 "asym_cpuadamwds|unsloth-ohbm0|ligerloss1" 96000 1
echo "M4_EXIT=$?"

echo "=== M5 llama T2 192k b2 (base 548 / 171.1) $(date -u +%H:%M:%S)"
env $POL $T2D MAX_SAMPLES=1024 bash scripts/lf/tp_probe.sh llama3.3-70b cpum5 "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 192000 2
echo "M5_EXIT=$?"

echo "=== M7 moe KA-dial 120k b8 (base 2762 / 165.7) $(date -u +%H:%M:%S)"
env $POL ASYM_GEMM_DISPATCH=staged ASYMM_QWEN3_MOE_FG_KEEP_ACTS_HBM=1 ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024 ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1 MAX_SAMPLES=1024 bash scripts/lf/tp_probe.sh q3-30b-a3b cpum7 "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 120000 8
echo "M7_EXIT=$?"

echo "=== M6 llama T2 448k b1 WALL (base 280 / 182.4) $(date -u +%H:%M:%S)"
env $POL $T2D MAX_SAMPLES=512 bash scripts/lf/tp_probe.sh llama3.3-70b cpum6 "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 448000 1
echo "M6_EXIT=$?"

echo "=== M3 q32 T3 640k b1 (base 219 / 129.7) $(date -u +%H:%M:%S)"
env $POL MAX_SAMPLES=512 bash scripts/lf/tp_probe.sh q3-32b cpum3 "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 640000 1
echo "M3_EXIT=$?"

echo "=== M8 moe shed 800k b1 (base 584 / 110.4) $(date -u +%H:%M:%S)"
env $POL ASYM_GEMM_DISPATCH=staged $PINS MAX_SAMPLES=1024 bash scripts/lf/tp_probe.sh q3-30b-a3b cpum8 "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 800000 1
echo "M8_EXIT=$?"

echo "=== M9 moe shed 1.1M b1 (base 385 / 152.9) $(date -u +%H:%M:%S)"
env $POL ASYM_GEMM_DISPATCH=staged $PINS MAX_SAMPLES=512 bash scripts/lf/tp_probe.sh q3-30b-a3b cpum9 "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 1100000 1
echo "M9_EXIT=$?"
echo "=== CPU-Q3 DONE $(date -u +%H:%M:%S)"
