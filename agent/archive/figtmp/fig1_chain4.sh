#!/bin/bash
# FIG-1b correction #3 (INSIDE container): MoE pair at 192K (torch walls
# measured at 320K/416K/512K — all censored >=189 GiB; 32K som anchor 11.4
# GiB acts-class -> slope ~0.405/K -> 192K predicted ~137 GiB = 74%).
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

guard E1
echo "=== E1: torch q30b 192K b1 (w1+m1) $(date -u +%H:%M:%S)"
WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 \
  bash scripts/lf/tp_probe.sh q3-30b-a3b f1etorm "torch|unsloth-ohbm0|ligerloss1" 192000 1
echo "F1_E1_EXIT=$?"

guard E2
echo "=== E2: z3om q30b 192K b1 (w1+m1) $(date -u +%H:%M:%S)"
WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 \
  bash scripts/lf/tp_probe.sh q3-30b-a3b f1ez3m "zero3_offload_mem|unsloth-ohbm0|ligerloss1" 192000 1
echo "F1_E2_EXIT=$?"

echo "=== FIG1 CHAIN4 DONE $(date -u +%H:%M:%S)"
