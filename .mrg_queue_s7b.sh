#!/bin/bash
# Queue 4: the three llama S7 rows (re-run after weights download; HF_TOKEN
# exported by the launcher). Envs verbatim from archived c12 runs.
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
T2ENV="ASYM_GEMM_DISPATCH=staged ASYMM_DENSE_MLP_FG_KEEP_ACTS_HBM=1 ASYM_SAVED_TENSOR_ASYNC_UNPACK=1 ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1 ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024"

echo "=== S7 ROW4 llama T1 96k b1 (ref 1066/48.9/486) $(date -u +%H:%M:%S)"
MAX_SAMPLES=512 ASYM_GEMM_DISPATCH=staged bash scripts/lf/tp_probe.sh llama3.3-70b mrgs7r4 "asym_cpuadamwds|unsloth-ohbm0|ligerloss1" 96000 1
echo "R4_EXIT=$?"

echo "=== S7 ROW5 llama T2 192k b2 (ref 543/171.1/963) $(date -u +%H:%M:%S)"
env $T2ENV MAX_SAMPLES=1024 bash scripts/lf/tp_probe.sh llama3.3-70b mrgs7r5 "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 192000 2
echo "R5_EXIT=$?"

echo "=== S7 ROW6 llama T2 448k b1 WALL (ref 275/182.4/983) $(date -u +%H:%M:%S)"
env $T2ENV MAX_SAMPLES=512 bash scripts/lf/tp_probe.sh llama3.3-70b mrgs7r6 "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 448000 1
echo "R6_EXIT=$?"
echo "=== QUEUE-4 DONE $(date -u +%H:%M:%S)"
