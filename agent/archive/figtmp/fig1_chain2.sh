#!/bin/bash
# FIG-1b correction chain (INSIDE container): the final pair configs/lengths.
# Pair = torch|unsloth (No Offload) vs zero3_offload_mem|unsloth (model states
# off, activations untouched — uns-off was disqualified: it offloads the GC
# checkpoints so the activation segment would not match across the pair).
# Lengths recalibrated from the measured walls: 32B 224K->160K (~153 GiB
# pred), MoE 512K->416K (~164 pred). All w1+m1 memory probes.
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

guard C1
echo "=== C1: torch q32 160K b1 (w1+m1) $(date -u +%H:%M:%S)"
WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 \
  bash scripts/lf/tp_probe.sh q3-32b f1ctorq "torch|unsloth-ohbm0|ligerloss1" 160000 1
echo "F1_C1_EXIT=$?"

guard C2
echo "=== C2: z3om q32 160K b1 (w1+m1) $(date -u +%H:%M:%S)"
WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 \
  bash scripts/lf/tp_probe.sh q3-32b f1cz3q "zero3_offload_mem|unsloth-ohbm0|ligerloss1" 160000 1
echo "F1_C2_EXIT=$?"

guard C3
echo "=== C3: torch q30b 416K b1 (w1+m1) $(date -u +%H:%M:%S)"
WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 \
  bash scripts/lf/tp_probe.sh q3-30b-a3b f1ctorm "torch|unsloth-ohbm0|ligerloss1" 416000 1
echo "F1_C3_EXIT=$?"

guard C4
echo "=== C4: z3om q30b 416K b1 (w1+m1) $(date -u +%H:%M:%S)"
WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 \
  bash scripts/lf/tp_probe.sh q3-30b-a3b f1cz3m "zero3_offload_mem|unsloth-ohbm0|ligerloss1" 416000 1
echo "F1_C4_EXIT=$?"

echo "=== FIG1 CHAIN2 DONE $(date -u +%H:%M:%S)"
