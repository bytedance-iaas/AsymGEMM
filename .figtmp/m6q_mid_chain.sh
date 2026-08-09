#!/bin/bash
# Qwen3-30B-A3B fig-3 short-column middle-length hunt (INSIDE container).
# Kevin 2026-08-05: need T1 meaningfully > T2 in tok/s, absolute scale
# ~1000-2000 (not 5k+ dwarfing the 1.1M ~387 bars). Trio at 256K b1, timed
# w1+m2, same presets as the banked row (T1 = moe|T1, T2 = moe|T2B, T3 =
# moe|T3 with ker101 fine on qwen). Fresh tags m6qm*.
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
echo "=== Q1: T1 qwen 256K b1 (w1+m2) $(date -u +%H:%M:%S)"
MAX_SAMPLES=512 bash scripts/lf/tp_probe.sh q3-30b-a3b m6qmt1256 "asym_cpuadamwds|T1|ligerloss1" 256000 1
echo "M6QM_T1_EXIT=$?"

guard Q2
echo "=== Q2: T2B qwen 256K b1 (w1+m2) $(date -u +%H:%M:%S)"
MAX_SAMPLES=512 bash scripts/lf/tp_probe.sh q3-30b-a3b m6qmt2b256 "asym_cpuadamwds|T2B|ligerloss1" 256000 1
echo "M6QM_T2B_EXIT=$?"

guard Q3
echo "=== Q3: T3 qwen 256K b1 (w1+m2) $(date -u +%H:%M:%S)"
MAX_SAMPLES=512 bash scripts/lf/tp_probe.sh q3-30b-a3b m6qmt3256 "asym_cpuadamwds|T3|ligerloss1" 256000 1
echo "M6QM_T3_EXIT=$?"
echo "=== QWEN MID CHAIN DONE $(date -u +%H:%M:%S)"
