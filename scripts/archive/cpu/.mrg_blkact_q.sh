#!/bin/bash
# toconfirm #1 salvage validation: blocked CPU-act (ASYMM_QWEN3_MOE_FG_BLOCKED_CPU_ACT).
# All runs: policy ON (production stack) + full pins (blocked path engages).
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
POL="ASYM_PLACEMENT_POLICY=1 ASYM_CPU_OPS_THREADS=48"
PINS="ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU=1 ASYMM_QWEN3_MOE_FG_DA_GPU=1 ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0 ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024 ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1 ASYMM_QWEN3_MOE_FG_KEEP_DGRADS_HBM=1"

echo "=== BA.1 SMOKE OFF (8k b4 m6 fixed seed, pins) $(date -u +%H:%M:%S)"
env $POL $PINS PROFILERS=source MAX_STEPS=6 WARMUP_STEPS=1 MAX_SAMPLES=1024 OVERWRITE=true RUN_NAME=basmk0 \
  RUNS="q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1 ; 8000|4|1 ; none|false|false|false|false|false" \
  timeout --kill-after=60 3600 bash scripts/lf/profile_lora_lf_test_source.sh
echo "BA1_EXIT=$?"

echo "=== BA.2 SMOKE ON $(date -u +%H:%M:%S)"
env $POL $PINS ASYMM_QWEN3_MOE_FG_BLOCKED_CPU_ACT=1 PROFILERS=source MAX_STEPS=6 WARMUP_STEPS=1 MAX_SAMPLES=1024 OVERWRITE=true RUN_NAME=basmk1 \
  RUNS="q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1 ; 8000|4|1 ; none|false|false|false|false|false" \
  timeout --kill-after=60 3600 bash scripts/lf/profile_lora_lf_test_source.sh
echo "BA2_EXIT=$?"

echo "=== BA.3 A/B OFF 32k b8 (w1+m4) $(date -u +%H:%M:%S)"
env $POL $PINS PROFILERS=source MAX_STEPS=4 WARMUP_STEPS=1 MAX_SAMPLES=1024 OVERWRITE=true RUN_NAME=baab0 \
  RUNS="q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1 ; 32000|8|1 ; none|false|false|false|false|false" \
  timeout --kill-after=60 5400 bash scripts/lf/profile_lora_lf_test_source.sh
echo "BA3_EXIT=$?"

echo "=== BA.4 A/B ON 32k b8 $(date -u +%H:%M:%S)"
env $POL $PINS ASYMM_QWEN3_MOE_FG_BLOCKED_CPU_ACT=1 PROFILERS=source MAX_STEPS=4 WARMUP_STEPS=1 MAX_SAMPLES=1024 OVERWRITE=true RUN_NAME=baab1 \
  RUNS="q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1 ; 32000|8|1 ; none|false|false|false|false|false" \
  timeout --kill-after=60 5400 bash scripts/lf/profile_lora_lf_test_source.sh
echo "BA4_EXIT=$?"
echo "=== BLKACT-Q DONE $(date -u +%H:%M:%S)"
