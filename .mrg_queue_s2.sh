#!/bin/bash
# Queue 2: S1-V4 (bundle delivery, both drivers) -> S4-V2 pair -> S4-V3 -> C4 -> C4b
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
BUNDLE="ASYM_GEMM_DISPATCH=staged ASYMM_QWEN3_MOE_FG_KEEP_ACTS_HBM=1 ASYMM_ATTN_ACT_KEEP_ACTS_HBM=1 ASYM_GC_SAVE_ON_CPU_OVERRIDE=false ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU=1 ASYMM_QWEN3_MOE_FG_DA_GPU=1 ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0 ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024 ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1"

echo "=== QUEUE V4a (tp_probe/_source, bundle 120k b4) $(date -u +%H:%M:%S)"
env $BUNDLE bash scripts/lf/tp_probe.sh q3-30b-a3b mrgv4a "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 120000 4
echo "V4A_EXIT=$?"

echo "=== QUEUE V4b (_both driver, bundle 120k b4) $(date -u +%H:%M:%S)"
env $BUNDLE PROFILERS=source MAX_STEPS=2 WARMUP_STEPS=1 MAX_SAMPLES=1024 RUN_NAME=mrgv4b \
  RUNS="q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1 ; 120000|4|1 ; none|false|false|false|false|false" \
  timeout --kill-after=60 5400 bash scripts/lf/profile_lora_lf_test_both.sh
echo "V4B_EXIT=$?"

echo "=== QUEUE S4-V2a (|T2| preset, 24k b8) $(date -u +%H:%M:%S)"
PROFILERS=source MAX_STEPS=2 WARMUP_STEPS=1 MAX_SAMPLES=1024 RUN_NAME=mrgv2t \
  RUNS="q3-30b-a3b|1 ; asym_cpuadamwds|T2|ligerloss1 ; 24000|8|1 ; none|false|false|false|false|false" \
  timeout --kill-after=60 3600 bash scripts/lf/profile_lora_lf_test_source.sh
echo "V2A_EXIT=$?"

echo "=== QUEUE S4-V2b (raw token no-KA, 24k b8) $(date -u +%H:%M:%S)"
PROFILERS=source MAX_STEPS=2 WARMUP_STEPS=1 MAX_SAMPLES=1024 RUN_NAME=mrgv2t \
  RUNS="q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1 ; 24000|8|1 ; none|false|false|false|false|false" \
  timeout --kill-after=60 3600 bash scripts/lf/profile_lora_lf_test_source.sh
echo "V2B_EXIT=$?"

echo "=== QUEUE C4 (900k bundle record reproduction) $(date -u +%H:%M:%S)"
env $BUNDLE MAX_SAMPLES=512 bash scripts/lf/tp_probe.sh q3-30b-a3b mrgc4 "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 900000 1
echo "C4_EXIT=$?"

echo "=== QUEUE C4b (800k shed record reproduction) $(date -u +%H:%M:%S)"
ASYM_GEMM_DISPATCH=staged ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU=1 ASYMM_QWEN3_MOE_FG_DA_GPU=1 ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0 ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024 ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1 \
  bash scripts/lf/tp_probe.sh q3-30b-a3b mrgc4b "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 800000 1
echo "C4B_EXIT=$?"
echo "=== QUEUE-2 DONE $(date -u +%H:%M:%S)"
