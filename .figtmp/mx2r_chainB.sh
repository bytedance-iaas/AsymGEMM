#!/bin/bash
# MIXTRAL 2-RANK campaign, chain B (INSIDE container):
#  1. rc/un @32k b2/b1 — close the batch-rescue question for the SO family
#  2. zero3_offload_mem|recomp @32k/@64k — the leaner baseline (series
#     "ZeRO3 Offload" = same recompute strategy as rc per the emitter note)
#  3. asym T1 ladder @128k/@192k
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
export GPU_POOL=0,1
export ASYM_ARENA_SHM_CAP_GB=300

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

P=".figtmp/tp2_probe.sh"
run() {
  local tag="$1" cfg="$2" seq="$3"; shift 3
  guard "$tag"
  echo "=== $tag @$seq b-walk: $* $(date -u +%H:%M:%S)"
  MAX_SAMPLES=512 bash "$P" mixtral-8x22b "$tag" "$cfg" "$seq" "$@"
  echo "MX2R_${tag}_EXIT=$?"
}

run mx2rc32b "superoffload_mem|recomp|ligerloss1"  32000 2 1
run mx2un32b "superoffload_mem|unsloth|ligerloss1" 32000 2 1
run mx2z332  "zero3_offload_mem|recomp|ligerloss1" 32000 4 2 1
run mx2z364  "zero3_offload_mem|recomp|ligerloss1" 64000 2 1
run mx2t1128 "asym_sdp2_cpuadamwds|T1|ligerloss1"  128000 2 1
run mx2t1192 "asym_sdp2_cpuadamwds|T1|ligerloss1"  192000 2 1
echo "=== MX2R CHAIN-B DONE $(date -u +%H:%M:%S)"
