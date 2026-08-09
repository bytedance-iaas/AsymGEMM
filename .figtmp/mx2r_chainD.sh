#!/bin/bash
# MIXTRAL 2-RANK chain D: tier promotion retry — T2B's fg engine banks ~307 GB
# in the fabric (chain-C hardfail: cap exceeded by 1.6 GB at 00408:bank).
# Cap 285 -> 340. shm hygiene per run.
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
export GPU_POOL=0,1
export ASYM_ARENA_SHM_CAP_GB=340
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
run() {
  local tag="$1" cfg="$2" seq="$3"; shift 3
  guard "$tag"
  echo "=== $tag @$seq b-walk: $* $(date -u +%H:%M:%S)"
  MAX_SAMPLES=512 bash "$P" mixtral-8x22b "$tag" "$cfg" "$seq" "$@"
  echo "MX2R_${tag}_EXIT=$?"
}
set +e
run mx2t2320b "asym_sdp2_cpuadamwds|T2B|ligerloss1" 320000 1
run mx2t2352b "asym_sdp2_cpuadamwds|T2B|ligerloss1" 352000 1
echo "=== MX2R CHAIN-D DONE $(date -u +%H:%M:%S)"
