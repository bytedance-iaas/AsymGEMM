#!/bin/bash
# CPU-merge queue 2: S4 module validation — SMOKE parity + donor-regime A/Bs.
# Donor protocol for A/Bs: w1+m4, steady = mean of middle 2 (matches donor numbers).
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
POL="ASYM_PLACEMENT_POLICY=1 ASYM_CPU_OPS_THREADS=48"

echo "=== S4.1a SMOKE parity OFF (moe 8k b4 m6, fixed seed) $(date -u +%H:%M:%S)"
PROFILERS=source MAX_STEPS=6 WARMUP_STEPS=1 MAX_SAMPLES=1024 OVERWRITE=true RUN_NAME=cpusmk0 \
  RUNS="q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker101-ceil0000-ohbm0|ligerloss1 ; 8000|4|1 ; none|false|false|false|false|false" \
  timeout --kill-after=60 3600 bash scripts/lf/profile_lora_lf_test_source.sh
echo "SMK0_EXIT=$?"

echo "=== S4.1b SMOKE parity ON (same, policy armed) $(date -u +%H:%M:%S)"
env $POL PROFILERS=source MAX_STEPS=6 WARMUP_STEPS=1 MAX_SAMPLES=1024 OVERWRITE=true RUN_NAME=cpusmk1 \
  RUNS="q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker101-ceil0000-ohbm0|ligerloss1 ; 8000|4|1 ; none|false|false|false|false|false" \
  timeout --kill-after=60 3600 bash scripts/lf/profile_lora_lf_test_source.sh
echo "SMK1_EXIT=$?"

echo "=== S4.2a-OFF MoE 32k b8 donor cfg (w1+m4) $(date -u +%H:%M:%S)"
PROFILERS=source MAX_STEPS=4 WARMUP_STEPS=1 MAX_SAMPLES=1024 OVERWRITE=true RUN_NAME=cpuab32off \
  RUNS="q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker101-ceil0000-ohbm0|ligerloss1 ; 32000|8|1 ; none|false|false|false|false|false" \
  timeout --kill-after=60 5400 bash scripts/lf/profile_lora_lf_test_source.sh
echo "AB32OFF_EXIT=$?"

echo "=== S4.2a-ON MoE 32k b8 + policy $(date -u +%H:%M:%S)"
env $POL PROFILERS=source MAX_STEPS=4 WARMUP_STEPS=1 MAX_SAMPLES=1024 OVERWRITE=true RUN_NAME=cpuab32on \
  RUNS="q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker101-ceil0000-ohbm0|ligerloss1 ; 32000|8|1 ; none|false|false|false|false|false" \
  timeout --kill-after=60 5400 bash scripts/lf/profile_lora_lf_test_source.sh
echo "AB32ON_EXIT=$?"

echo "=== S4.2c-OFF dense 32B 32k b8 donor cfg $(date -u +%H:%M:%S)"
PROFILERS=source MAX_STEPS=4 WARMUP_STEPS=1 MAX_SAMPLES=1024 OVERWRITE=true RUN_NAME=cpuabd32off \
  RUNS="q3-32b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm8|ligerloss1 ; 32000|8|1 ; none|false|false|false|false|false" \
  timeout --kill-after=60 5400 bash scripts/lf/profile_lora_lf_test_source.sh
echo "ABD32OFF_EXIT=$?"

echo "=== S4.2c-ON dense 32B 32k b8 + policy $(date -u +%H:%M:%S)"
env $POL PROFILERS=source MAX_STEPS=4 WARMUP_STEPS=1 MAX_SAMPLES=1024 OVERWRITE=true RUN_NAME=cpuabd32on \
  RUNS="q3-32b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm8|ligerloss1 ; 32000|8|1 ; none|false|false|false|false|false" \
  timeout --kill-after=60 5400 bash scripts/lf/profile_lora_lf_test_source.sh
echo "ABD32ON_EXIT=$?"
echo "=== CPU-Q2 DONE $(date -u +%H:%M:%S)"
