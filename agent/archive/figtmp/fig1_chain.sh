#!/bin/bash
# FIG-1 real-value chain (INSIDE container). 8 runs, serial, guards.
# 1a (optimizer share; token unsloth; 32K b1; w1+m2): zero3_offload_mem vs
#    superoffload_mem, Qwen3-32B + Qwen3-30B-A3B.
# 1b (peak-memory breakdown; w1+m1 memory probes): torch|unsloth (No Offload)
#    vs superoffload_mem|unsloth-off (SuperOffload), 32B@224K b1, MoE@512K b1.
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM

guard() {
  sleep 30
  local apps
  for _ in $(seq 1 21); do
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
    [ -z "$apps" ] && return 0
    echo "!!! GUARD: GPU busy before $1 (pids: $apps) — waiting"
    sleep 30
  done
  echo "!!! GUARD FAIL before $1 — aborting chain"
  exit 9
}

echo "=== A1: z3om q32 32K b1 (w1+m2) $(date -u +%H:%M:%S)"
bash scripts/lf/tp_probe.sh q3-32b f1az3q "zero3_offload_mem|unsloth-ohbm0|ligerloss1" 32000 1
echo "F1_A1_EXIT=$?"

guard A2
echo "=== A2: som q32 32K b1 (w1+m2) $(date -u +%H:%M:%S)"
bash scripts/lf/tp_probe.sh q3-32b f1asoq "superoffload_mem|unsloth-ohbm0|ligerloss1" 32000 1
echo "F1_A2_EXIT=$?"

guard A3
echo "=== A3: z3om q30b 32K b1 (w1+m2) $(date -u +%H:%M:%S)"
bash scripts/lf/tp_probe.sh q3-30b-a3b f1az3m "zero3_offload_mem|unsloth-ohbm0|ligerloss1" 32000 1
echo "F1_A3_EXIT=$?"

guard A4
echo "=== A4: som q30b 32K b1 (w1+m2) $(date -u +%H:%M:%S)"
bash scripts/lf/tp_probe.sh q3-30b-a3b f1asom "superoffload_mem|unsloth-ohbm0|ligerloss1" 32000 1
echo "F1_A4_EXIT=$?"

guard B1
echo "=== B1: torch q32 224K b1 (w1+m1 memory) $(date -u +%H:%M:%S)"
WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 \
  bash scripts/lf/tp_probe.sh q3-32b f1btorq "torch|unsloth-ohbm0|ligerloss1" 224000 1
echo "F1_B1_EXIT=$?"

guard B2
echo "=== B2: so|uns-off q32 224K b1 (w1+m1 memory) $(date -u +%H:%M:%S)"
WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 \
  bash scripts/lf/tp_probe.sh q3-32b f1buoq "superoffload_mem|unsloth-off-ohbm0|ligerloss1" 224000 1
echo "F1_B2_EXIT=$?"

guard B3
echo "=== B3: torch q30b 512K b1 (w1+m1 memory) $(date -u +%H:%M:%S)"
WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 \
  bash scripts/lf/tp_probe.sh q3-30b-a3b f1btorm "torch|unsloth-ohbm0|ligerloss1" 512000 1
echo "F1_B3_EXIT=$?"

guard B4
echo "=== B4: so|uns-off q30b 512K b1 (w1+m1 memory) $(date -u +%H:%M:%S)"
WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 \
  bash scripts/lf/tp_probe.sh q3-30b-a3b f1buom "superoffload_mem|unsloth-off-ohbm0|ligerloss1" 512000 1
echo "F1_B4_EXIT=$?"

echo "=== FIG1 CHAIN DONE $(date -u +%H:%M:%S)"
