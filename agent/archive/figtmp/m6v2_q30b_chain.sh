#!/bin/bash
# m6v2 q3-30b-a3b chain (runs INSIDE container): T2B@1.4M OOM-attempt, then the
# three timed cells. Strictly serial, 30 s settle, GPU-idle guard between steps.
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM

guard() {
  sleep 30
  local apps
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
  if [ -n "$apps" ]; then
    echo "!!! GUARD: GPU not idle before $1 (pids: $apps) — waiting up to 10 min"
    for _ in $(seq 1 20); do
      sleep 30
      apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
      [ -z "$apps" ] && break
    done
    [ -n "$apps" ] && { echo "!!! GUARD FAIL before $1 — aborting chain"; exit 9; }
  fi
}

echo "=== C: T2B@1.4M OOM-attempt (builds 1.4M n512 dataset) $(date -u +%H:%M:%S)"
WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 \
  bash scripts/lf/tp_probe.sh q3-30b-a3b m6v2t2b1400 "asym_cpuadamwds|T2B|ligerloss1" 1400000 1
echo "M6V2_T2B1400_EXIT=$?"

guard "D"
echo "=== D: T2B@800K timed w1+m2 (n1024) $(date -u +%H:%M:%S)"
bash scripts/lf/tp_probe.sh q3-30b-a3b m6v2t2b800 "asym_cpuadamwds|T2B|ligerloss1" 800000 1
echo "M6V2_T2B800_EXIT=$?"

guard "E"
echo "=== E: T3@800K timed w1+m2 (n1024) $(date -u +%H:%M:%S)"
bash scripts/lf/tp_probe.sh q3-30b-a3b m6v2t3800 "asym_cpuadamwds|T3|ligerloss1" 800000 1
echo "M6V2_T3800_EXIT=$?"

guard "F"
echo "=== F: T3@1.1M timed w1+m2 (n512) $(date -u +%H:%M:%S)"
MAX_SAMPLES=512 bash scripts/lf/tp_probe.sh q3-30b-a3b m6v2t31100 "asym_cpuadamwds|T3|ligerloss1" 1100000 1
echo "M6V2_T31100_EXIT=$?"

echo "=== Q30B CHAIN DONE $(date -u +%H:%M:%S)"
