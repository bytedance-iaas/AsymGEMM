#!/bin/bash
# FIG-1a q30b @128K pair (INSIDE container): z3om + som, w1+m2 — uniform
# 128K length across both 1a groups. Launch AFTER the mixtral pair chain.
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

guard Q1
echo "=== Q1: z3om q30b 128K b1 (w1+m2) $(date -u +%H:%M:%S)"
bash scripts/lf/tp_probe.sh q3-30b-a3b f1bz3q2 "zero3_offload_mem|unsloth-ohbm0|ligerloss1" 128000 1
echo "F1A_Q1_EXIT=$?"

guard Q2
echo "=== Q2: som q30b 128K b1 (w1+m2) $(date -u +%H:%M:%S)"
bash scripts/lf/tp_probe.sh q3-30b-a3b f1bsoq2 "superoffload_mem|unsloth-ohbm0|ligerloss1" 128000 1
echo "F1A_Q2_EXIT=$?"

echo "=== FIG1A Q30B PAIR DONE $(date -u +%H:%M:%S)"
