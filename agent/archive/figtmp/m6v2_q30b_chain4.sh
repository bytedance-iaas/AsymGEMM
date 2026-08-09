#!/bin/bash
# m6v2 q3-30b-a3b chain4 (INSIDE container): the 80K b8 short column
# (T1 re-verify for the memory panel + T2B + T3). Serial, guards.
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

guard A
echo "=== A: T1@80K b8 timed w1+m2 (n1024) $(date -u +%H:%M:%S)"
bash scripts/lf/tp_probe.sh q3-30b-a3b m6v2t180 "asym_cpuadamwds|T1|ligerloss1" 80000 8
echo "M6V2_T180_EXIT=$?"

guard B
echo "=== B: T2B@80K b8 timed w1+m2 (n1024) $(date -u +%H:%M:%S)"
bash scripts/lf/tp_probe.sh q3-30b-a3b m6v2t2b80 "asym_cpuadamwds|T2B|ligerloss1" 80000 8
echo "M6V2_T2B80_EXIT=$?"

guard C
echo "=== C: T3@80K b8 timed w1+m2 (n1024) $(date -u +%H:%M:%S)"
bash scripts/lf/tp_probe.sh q3-30b-a3b m6v2t380 "asym_cpuadamwds|T3|ligerloss1" 80000 8
echo "M6V2_T380_EXIT=$?"

guard D
echo "=== D: T2B@1.1M timed w1+m2 (n512, same-day pair vs m6v2t31100) $(date -u +%H:%M:%S)"
MAX_SAMPLES=512 bash scripts/lf/tp_probe.sh q3-30b-a3b m6v2t2b1100b "asym_cpuadamwds|T2B|ligerloss1" 1100000 1
echo "M6V2_T2B1100B_EXIT=$?"

echo "=== Q30B CHAIN4 DONE $(date -u +%H:%M:%S)"
