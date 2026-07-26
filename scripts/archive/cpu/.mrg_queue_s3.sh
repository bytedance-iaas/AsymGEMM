#!/bin/bash
# Queue 3: C4'/C4b' reruns WITH the 6th pin (KEEP_DGRADS_HBM=1), then all S7 rows.
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
PINS6="ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU=1 ASYMM_QWEN3_MOE_FG_DA_GPU=1 ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0 ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024 ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1 ASYMM_QWEN3_MOE_FG_KEEP_DGRADS_HBM=1"
T2ENV="ASYM_GEMM_DISPATCH=staged ASYMM_DENSE_MLP_FG_KEEP_ACTS_HBM=1 ASYM_SAVED_TENSOR_ASYNC_UNPACK=1 ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1 ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024"

echo "=== C4b' 800k shed + dgrads (ref 596/147.5/539) $(date -u +%H:%M:%S)"
env ASYM_GEMM_DISPATCH=staged $PINS6 bash scripts/lf/tp_probe.sh q3-30b-a3b mrgc4bd "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 800000 1
echo "C4BD_EXIT=$?"

echo "=== C4' 900k bundle + dgrads (ref 519/183.0) $(date -u +%H:%M:%S)"
env ASYM_GEMM_DISPATCH=staged ASYMM_QWEN3_MOE_FG_KEEP_ACTS_HBM=1 ASYMM_ATTN_ACT_KEEP_ACTS_HBM=1 ASYM_GC_SAVE_ON_CPU_OVERRIDE=false $PINS6 MAX_SAMPLES=512 bash scripts/lf/tp_probe.sh q3-30b-a3b mrgc4d "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 900000 1
echo "C4D_EXIT=$?"

echo "=== S7 ROW4 llama T1 96k b1 (ref 1066/48.9/486) $(date -u +%H:%M:%S)"
MAX_SAMPLES=512 ASYM_GEMM_DISPATCH=staged bash scripts/lf/tp_probe.sh llama3.3-70b mrgs7r4 "asym_cpuadamwds|unsloth-ohbm0|ligerloss1" 96000 1
echo "R4_EXIT=$?"

echo "=== S7 ROW5 llama T2 192k b2 (ref 543/171.1/963) $(date -u +%H:%M:%S)"
env $T2ENV MAX_SAMPLES=1024 bash scripts/lf/tp_probe.sh llama3.3-70b mrgs7r5 "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 192000 2
echo "R5_EXIT=$?"

echo "=== S7 ROW6 llama T2 448k b1 WALL (ref 275/182.4/983) $(date -u +%H:%M:%S)"
env $T2ENV MAX_SAMPLES=512 bash scripts/lf/tp_probe.sh llama3.3-70b mrgs7r6 "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 448000 1
echo "R6_EXIT=$?"

echo "=== S7 ROW3 q32 T3 640k b1 (ref 226/129.7/980; bare fg NO staged) $(date -u +%H:%M:%S)"
MAX_SAMPLES=512 bash scripts/lf/tp_probe.sh q3-32b mrgs7r3 "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 640000 1
echo "R3_EXIT=$?"

echo "=== S7 ROW9 moe 1.1M b1 shed (ref 382/151.5/906) $(date -u +%H:%M:%S)"
env ASYM_GEMM_DISPATCH=staged $PINS6 MAX_SAMPLES=512 bash scripts/lf/tp_probe.sh q3-30b-a3b mrgs7r9 "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 1100000 1
echo "R9_EXIT=$?"
echo "=== QUEUE-3 DONE $(date -u +%H:%M:%S)"
