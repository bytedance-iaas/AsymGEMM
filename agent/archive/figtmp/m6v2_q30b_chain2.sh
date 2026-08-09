#!/bin/bash
# m6v2 q3-30b-a3b chain2 REV2 (INSIDE container). Plotted trio = moe|T1
# (plain: unsloth token + staged) / T2B / T3 at 800K / 1.1M / 1.4M.
# Wall attempts first (lock the lengths), then timed cells. Serial,
# 30 s settle + GPU-idle guard between runs.
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

echo "=== A: T1plain@1.1M OOM-attempt (w1+m1, n512) $(date -u +%H:%M:%S)"
WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 \
  bash scripts/lf/tp_probe.sh q3-30b-a3b m6v2t11100 "asym_cpuadamwds|T1|ligerloss1" 1100000 1
echo "M6V2_T11100_EXIT=$?"

guard B
echo "=== B: T2B@1.4M OOM-attempt (w1+m1, builds s1400000 n512) $(date -u +%H:%M:%S)"
WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 \
  bash scripts/lf/tp_probe.sh q3-30b-a3b m6v2t2b1400 "asym_cpuadamwds|T2B|ligerloss1" 1400000 1
echo "M6V2_T2B1400_EXIT=$?"

guard C
echo "=== C: T1plain@800K timed w1+m2 (n1024) $(date -u +%H:%M:%S)"
bash scripts/lf/tp_probe.sh q3-30b-a3b m6v2t1800 "asym_cpuadamwds|T1|ligerloss1" 800000 1
echo "M6V2_T1800_EXIT=$?"

guard D
echo "=== D: T2B@800K timed w1+m2 (n1024) $(date -u +%H:%M:%S)"
bash scripts/lf/tp_probe.sh q3-30b-a3b m6v2t2b800 "asym_cpuadamwds|T2B|ligerloss1" 800000 1
echo "M6V2_T2B800_EXIT=$?"

guard E
echo "=== E: T3@800K timed w1+m2 (n1024) $(date -u +%H:%M:%S)"
bash scripts/lf/tp_probe.sh q3-30b-a3b m6v2t3800 "asym_cpuadamwds|T3|ligerloss1" 800000 1
echo "M6V2_T3800_EXIT=$?"

guard F
echo "=== F: T2B@1.1M timed w1+m2 (n512) $(date -u +%H:%M:%S)"
MAX_SAMPLES=512 bash scripts/lf/tp_probe.sh q3-30b-a3b m6v2t2b1100 "asym_cpuadamwds|T2B|ligerloss1" 1100000 1
echo "M6V2_T2B1100_EXIT=$?"

guard G
echo "=== G: T3@1.1M timed w1+m2 (n512) $(date -u +%H:%M:%S)"
MAX_SAMPLES=512 bash scripts/lf/tp_probe.sh q3-30b-a3b m6v2t31100 "asym_cpuadamwds|T3|ligerloss1" 1100000 1
echo "M6V2_T31100_EXIT=$?"

guard H
echo "=== H: T3@1.4M timed w1+m2 (n512) $(date -u +%H:%M:%S)"
MAX_SAMPLES=512 bash scripts/lf/tp_probe.sh q3-30b-a3b m6v2t31400 "asym_cpuadamwds|T3|ligerloss1" 1400000 1
echo "M6V2_T31400_EXIT=$?"

echo "=== Q30B CHAIN2 DONE $(date -u +%H:%M:%S)"
