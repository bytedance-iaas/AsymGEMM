#!/bin/bash
# MIXTRAL 2-RANK chain K — ceiling tighten: T1 @304K (between the 288K 98%
# edge-fit and the 320K measured G-OOM). Same run-form as the asym ladder
# (sdp2, arena 285, default watchdog floor), fused checkpoint.
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
echo "=== mx2t1304: T1 @304K b1 (w1+m1 capacity) $(date -u +%H:%M:%S)"
WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 \
  bash .figtmp/tp2_probe.sh mixtral-8x22b mx2t1304 "asym_sdp2_cpuadamwds|T1|ligerloss1" 304000 1
echo "MX2K_304_EXIT=$?"
echo "=== CHAIN-K DONE $(date -u +%H:%M:%S)"
