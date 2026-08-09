#!/bin/bash
# MIXTRAL 2-RANK campaign, chain C (INSIDE container): the upper asym ladder.
# SHM HYGIENE (chain-B lesson): the fabric file (/dev/shm/asym_fabric_<port>,
# pre-sized to ASYM_ARENA_SHM_CAP_GB) is never unlinked on exit; shm is only
# 479 GB -> one leaked file starves the next run into SIGBUS. rm before every
# probe. Cap trimmed 300->285 (bank 271 + headroom).
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
    echo "!!! GUARD: GPU busy before $1 (pids: $apps) — waiting"
    sleep 30
  done
  echo "!!! GUARD FAIL before $1 — aborting chain"
  exit 9
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
run mx2t1192b "asym_sdp2_cpuadamwds|T1|ligerloss1"  192000 1
run mx2t1256  "asym_sdp2_cpuadamwds|T1|ligerloss1"  256000 1
run mx2t1320  "asym_sdp2_cpuadamwds|T1|ligerloss1"  320000 1
if [ "$(grep -c 'MX2R_mx2t1320_EXIT=0' /dev/null 2>/dev/null)" != "x" ]; then :; fi
run mx2t2320  "asym_sdp2_cpuadamwds|T2B|ligerloss1" 320000 1
run mx2t2352  "asym_sdp2_cpuadamwds|T2B|ligerloss1" 352000 1
echo "=== MX2R CHAIN-C DONE $(date -u +%H:%M:%S)"
