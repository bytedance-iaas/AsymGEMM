#!/bin/bash
# FIG-1a mixtral pair (INSIDE container): z3om + som @128K b1, r64 default,
# w1+m2 — the two timing bars for the mixtral group of the pair-format 1a.
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

guard M1
echo "=== M1: z3om mixtral 128K b1 (w1+m2) $(date -u +%H:%M:%S)"
MAX_SAMPLES=512 bash scripts/lf/tp_probe.sh mixtral-8x22b f1az3x "zero3_offload_mem|unsloth-ohbm0|ligerloss1" 128000 1
echo "F1A_M1_EXIT=$?"

guard M2
echo "=== M2: som mixtral 128K b1 (w1+m2) $(date -u +%H:%M:%S)"
MAX_SAMPLES=512 bash scripts/lf/tp_probe.sh mixtral-8x22b f1asox "superoffload_mem|unsloth-ohbm0|ligerloss1" 128000 1
echo "F1A_M2_EXIT=$?"

echo "=== FIG1A MIXTRAL PAIR DONE $(date -u +%H:%M:%S)"
