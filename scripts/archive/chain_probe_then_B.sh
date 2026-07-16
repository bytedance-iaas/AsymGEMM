#!/bin/bash
# Sequential chain (one experiment at a time, in-container):
#   A. fg101 discriminator: canonical 5-case probe order with the CUDA caching
#      allocator DISABLED. fg101 passing here (vs 0.170918 with caching on)
#      confirms device-memory recycling as the contamination channel (V3).
#   B. re-measure baseline B (superoffload_mem|unsloth-off) at 45k, then 80k,
#      post-merge, to replace the 2026-07-14 B numbers under the G1 ratios (B2).
set -uo pipefail

echo "########## SECTION A: probe, caching-allocator OFF ##########"
PYTORCH_NO_CUDA_MEMORY_CACHING=1 CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python scripts/testing/qwen35_fg_numeric_probe.py --qwen3 --tokens 655360 \
  2>&1 | tee /workspace/qwen35_local/probe_nocache_655360.log | grep -E "shapes:|out |out~plain|verdict"
echo "########## SECTION A DONE ##########"

common() {
  export GPU_POOL=0 PROFILERS=source PLOT=false OVERWRITE=true
}

echo "########## SECTION B1: baseline B @45k (2026-07-14 ref: lat 261.8, alloc 59.4, reserved 72.4) ##########"
common
RUNS='q3.5-35b-a3b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 45000|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=/workspace/qwen35_local/profiling_postmerge_B45 \
  bash scripts/lf/profile_lora_lf_test_both.sh
echo "########## SECTION B1 DONE ##########"

echo "########## SECTION B2: baseline B @80k (2026-07-14 ref: lat 359.7-360.5, alloc 100.4, reserved 103.0) ##########"
common
RUNS='q3.5-35b-a3b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=/workspace/qwen35_local/profiling_postmerge_B80 \
  bash scripts/lf/profile_lora_lf_test_both.sh
echo "########## SECTION B2 DONE ##########"
