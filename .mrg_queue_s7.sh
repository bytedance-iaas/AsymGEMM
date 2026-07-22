#!/bin/bash
# Queue 3 = S7 remaining rows (3,4,5,6,9). Rows 1/2/7 = C1/C2/C3 (done, PASS);
# row 8 = C4b (queue 2). All envs verbatim from the archived runs' command.txt.
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
T2ENV="ASYM_GEMM_DISPATCH=staged ASYMM_DENSE_MLP_FG_KEEP_ACTS_HBM=1 ASYM_SAVED_TENSOR_ASYNC_UNPACK=1 ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1 ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024"

echo "=== S7 ROW4 llama T1 96k b1 (ref 1066 tok/s / 48.9 GiB / RSS 486; tputw7) $(date -u +%H:%M:%S)"
MAX_SAMPLES=512 ASYM_GEMM_DISPATCH=staged bash scripts/lf/tp_probe.sh llama3.3-70b mrgs7r4 "asym_cpuadamwds|unsloth-ohbm0|ligerloss1" 96000 1
echo "R4_EXIT=$?"

echo "=== S7 ROW5 llama T2 192k b2 (ref 543 tok/s / 171.1 GiB / RSS 963; tputask0) $(date -u +%H:%M:%S)"
env $T2ENV MAX_SAMPLES=1024 bash scripts/lf/tp_probe.sh llama3.3-70b mrgs7r5 "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 192000 2
echo "R5_EXIT=$?"

echo "=== S7 ROW6 llama T2 448k b1 WALL (ref 275 tok/s / 182.4 GiB / RSS 983; tputw6) $(date -u +%H:%M:%S)"
env $T2ENV MAX_SAMPLES=512 bash scripts/lf/tp_probe.sh llama3.3-70b mrgs7r6 "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 448000 1
echo "R6_EXIT=$?"

echo "=== S7 ROW3 q32 T3 640k b1 (ref 226 tok/s / 129.7 GiB / RSS 980; tputw4 — bare fg, NO staged) $(date -u +%H:%M:%S)"
MAX_SAMPLES=512 bash scripts/lf/tp_probe.sh q3-32b mrgs7r3 "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 640000 1
echo "R3_EXIT=$?"

echo "=== S7 ROW9 moe 1.1M b1 shed (ref 382 tok/s / 151.5 GiB / RSS 906; tputschedb) $(date -u +%H:%M:%S)"
ASYM_GEMM_DISPATCH=staged ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU=1 ASYMM_QWEN3_MOE_FG_DA_GPU=1 ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0 ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024 ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1 ASYMM_QWEN3_MOE_FG_KEEP_DGRADS_HBM=1 MAX_SAMPLES=512 \
  bash scripts/lf/tp_probe.sh q3-30b-a3b mrgs7r9 "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 1100000 1
echo "R9_EXIT=$?"
echo "=== S7 QUEUE DONE $(date -u +%H:%M:%S)"
