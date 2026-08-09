#!/bin/bash
# m6 MIXTRAL ROW re-measure, phase A (INSIDE container). Trio = the top row's
# presets: T1=moe|T1(unsloth+staged), T2=moe|T2B, T3=moe|T3. Col1=192K
# (T1 settled: f1settle192 992 tok/s / 128.9 resv). Here: the two remaining
# 192K bars (w1+m2) + the T1@256K capacity attempt (w1+m1) that decides col2.
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

guard A1
echo "=== A1: T2B mixtral 192K b1 (w1+m2) $(date -u +%H:%M:%S)"
MAX_SAMPLES=512 bash scripts/lf/tp_probe.sh mixtral-8x22b m6mxt2b192 "asym_cpuadamwds|T2B|ligerloss1" 192000 1
echo "M6MX_A1_EXIT=$?"

guard A2
echo "=== A2: T3 mixtral 192K b1 (w1+m2) $(date -u +%H:%M:%S)"
MAX_SAMPLES=512 bash scripts/lf/tp_probe.sh mixtral-8x22b m6mxt3192 "asym_cpuadamwds|T3|ligerloss1" 192000 1
echo "M6MX_A2_EXIT=$?"

guard A3
echo "=== A3: T1 mixtral 256K b1 OOM-attempt (w1+m1) $(date -u +%H:%M:%S)"
WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 \
  bash scripts/lf/tp_probe.sh mixtral-8x22b m6mxt1256 "asym_cpuadamwds|T1|ligerloss1" 256000 1
echo "M6MX_A3_EXIT=$?"

echo "=== M6MX CHAIN-A DONE $(date -u +%H:%M:%S)"
