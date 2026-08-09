#!/bin/bash
# FIG-1b correction #2 (INSIDE container): MoE pair at 320K (torch walls
# measured at 416K/512K: demand ~190; conservative slope 0.316/K -> 320K
# predicted ~160 GiB = 87% healthy). w1+m1 memory probes.
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

guard D1
echo "=== D1: torch q30b 320K b1 (w1+m1) $(date -u +%H:%M:%S)"
WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 \
  bash scripts/lf/tp_probe.sh q3-30b-a3b f1dtorm "torch|unsloth-ohbm0|ligerloss1" 320000 1
echo "F1_D1_EXIT=$?"

guard D2
echo "=== D2: z3om q30b 320K b1 (w1+m1) $(date -u +%H:%M:%S)"
WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 \
  bash scripts/lf/tp_probe.sh q3-30b-a3b f1dz3m "zero3_offload_mem|unsloth-ohbm0|ligerloss1" 320000 1
echo "F1_D2_EXIT=$?"

echo "=== FIG1 CHAIN3 DONE $(date -u +%H:%M:%S)"
