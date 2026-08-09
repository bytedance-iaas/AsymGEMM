#!/bin/bash
# MIXTRAL 2-RANK chain E: the 6th live column — T1 edge probe at 288k
# (T2B fabric bank ~540 GB > shm 479: tier promotion infeasible on this node;
# T1@320k = beyond-band 98.05% red). Fallback 272k if 288k lands beyond-band.
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
export GPU_POOL=0,1
export ASYM_ARENA_SHM_CAP_GB=285
guard() {
  sleep 30
  local apps
  for _ in $(seq 1 21); do
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
    [ -z "$apps" ] && { rm -f /dev/shm/asym_fabric_*; return 0; }
    echo "!!! GUARD: GPU busy before $1 (pids: $apps) — waiting"; sleep 30
  done
  echo "!!! GUARD FAIL before $1 — aborting chain"; exit 9
}
P=".figtmp/tp2_probe.sh"
set +e
guard T1_288
echo "=== mx2t1288 @288000 b-walk: 1 $(date -u +%H:%M:%S)"
MAX_SAMPLES=512 bash "$P" mixtral-8x22b mx2t1288 "asym_sdp2_cpuadamwds|T1|ligerloss1" 288000 1
E288=$?
echo "MX2R_mx2t1288_EXIT=$E288"
if [ $E288 -ne 0 ]; then
  guard T1_272
  echo "=== mx2t1272 @272000 b-walk: 1 $(date -u +%H:%M:%S)"
  MAX_SAMPLES=512 bash "$P" mixtral-8x22b mx2t1272 "asym_sdp2_cpuadamwds|T1|ligerloss1" 272000 1
  echo "MX2R_mx2t1272_EXIT=$?"
fi
echo "=== MX2R CHAIN-E DONE $(date -u +%H:%M:%S)"
