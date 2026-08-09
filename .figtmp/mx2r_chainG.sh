#!/bin/bash
# MIXTRAL 2-RANK chain G — fabric-off ablation: asym_dp2 (plain DP2, no shm
# bank) @32K. If it dies like the baselines, the panel gap is attributed to
# the shared bank itself, within our own system.
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
export GPU_POOL=0,1
export HOST_MEM_WATCHDOG_FLOOR_GB=35
sleep 30
for _ in $(seq 1 21); do
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
  [ -z "$apps" ] && { rm -f /dev/shm/asym_fabric_*; break; }
  echo "!!! GUARD: GPU busy (pids: $apps) — waiting"; sleep 30
done
echo "=== mx2gdp232: asym_dp2 (NO fabric) @32K b-walk 4 2 1 $(date -u +%H:%M:%S)"
MAX_SAMPLES=512 bash .figtmp/tp2_probe.sh mixtral-8x22b mx2gdp232 "asym_dp2_cpuadamwds|T1|ligerloss1" 32000 4 2 1
echo "MX2R_mx2gdp232_EXIT=$?"
echo "=== MX2R CHAIN-G DONE $(date -u +%H:%M:%S)"
