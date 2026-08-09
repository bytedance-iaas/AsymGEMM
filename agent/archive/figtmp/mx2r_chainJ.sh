#!/bin/bash
# MIXTRAL 2-RANK chain J — the cascade ladder (fused checkpoint, floor 35,
# CLEAN mx2i* tags — the mx2F* set lowercased into chain-F's dirs, lesson
# recorded). Walk every baseline to its true wall.
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
export GPU_POOL=0,1
export HOST_MEM_WATCHDOG_FLOOR_GB=35
guard() {
  sleep 30
  local apps
  for _ in $(seq 1 41); do
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
# batch catch-ups at short lengths
run mx2iz3a32  "zero3_offload_mem|recomp|ligerloss1"    32000 4
run mx2iz3a64  "zero3_offload_mem|recomp|ligerloss1"    64000 2 1
run mx2iuo64   "superoffload_mem|unsloth-off|ligerloss1" 64000 2 1
# 128k rung
run mx2irc128  "superoffload_mem|recomp|ligerloss1"      128000 1
run mx2iun128  "superoffload_mem|unsloth|ligerloss1"     128000 2 1
run mx2iuo128  "superoffload_mem|unsloth-off|ligerloss1" 128000 1
run mx2iz3128  "zero3_offload_mem|recomp|ligerloss1"     128000 1
# 192k rung (rc/z3 wall attempts expected)
run mx2irc192  "superoffload_mem|recomp|ligerloss1"      192000 1
run mx2iun192  "superoffload_mem|unsloth|ligerloss1"     192000 1
run mx2iuo192  "superoffload_mem|unsloth-off|ligerloss1" 192000 1
run mx2iz3192  "zero3_offload_mem|recomp|ligerloss1"     192000 1
# un/uo upper ladder to their walls
run mx2iun256  "superoffload_mem|unsloth|ligerloss1"     256000 1
run mx2iun288  "superoffload_mem|unsloth|ligerloss1"     288000 1
run mx2iun320  "superoffload_mem|unsloth|ligerloss1"     320000 1
echo "=== MX2R CHAIN-J DONE $(date -u +%H:%M:%S)"
