#!/bin/bash
# Chain L: timed w1+m2 mixtral 2r T1 @304K — panel-grade cell for the
# 288K->304K ceiling promotion (fresh tag; capacity probe mx2t1304 was w1+m1).
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
export GPU_POOL=0,1
export ASYM_ARENA_SHM_CAP_GB=285
sleep 30
for _ in $(seq 1 21); do
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
  [ -z "$apps" ] && { rm -f /dev/shm/asym_fabric_*; break; }
  echo "!!! GUARD: GPU busy (pids: $apps) — waiting"; sleep 30
done
echo "=== mx2t1304t: T1 @304K b1 TIMED (w1+m2) $(date -u +%H:%M:%S)"
MAX_SAMPLES=512 bash .figtmp/tp2_probe.sh mixtral-8x22b mx2t1304t "asym_sdp2_cpuadamwds|T1|ligerloss1" 304000 1
echo "MX2L_EXIT=$?"
