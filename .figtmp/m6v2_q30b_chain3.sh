#!/bin/bash
# m6v2 q3-30b-a3b chain3 (INSIDE container): the 3 remaining timed cells under
# full reuse of the c14 anchors. Serial, 30 s settle + GPU-idle guard.
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
echo "=== A: T2B@640K timed w1+m2 (n512) $(date -u +%H:%M:%S)"
MAX_SAMPLES=512 bash scripts/lf/tp_probe.sh q3-30b-a3b m6v2t2b640 "asym_cpuadamwds|T2B|ligerloss1" 640000 1
echo "M6V2_T2B640_EXIT=$?"

guard B
echo "=== B: T3@640K timed w1+m2 (n512) $(date -u +%H:%M:%S)"
MAX_SAMPLES=512 bash scripts/lf/tp_probe.sh q3-30b-a3b m6v2t3640 "asym_cpuadamwds|T3|ligerloss1" 640000 1
echo "M6V2_T3640_EXIT=$?"

guard C
echo "=== C: T3@1.1M timed w1+m2 (n512) $(date -u +%H:%M:%S)"
MAX_SAMPLES=512 bash scripts/lf/tp_probe.sh q3-30b-a3b m6v2t31100 "asym_cpuadamwds|T3|ligerloss1" 1100000 1
echo "M6V2_T31100_EXIT=$?"

guard D
echo "=== D: T2B@1.6M OOM-attempt (w1+m1, builds s1600000 n512) $(date -u +%H:%M:%S)"
WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 \
  bash scripts/lf/tp_probe.sh q3-30b-a3b m6v2t2b1600 "asym_cpuadamwds|T2B|ligerloss1" 1600000 1
echo "M6V2_T2B1600_EXIT=$?"

echo "=== Q30B CHAIN3 DONE $(date -u +%H:%M:%S)"
