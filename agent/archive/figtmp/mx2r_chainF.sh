#!/bin/bash
# MIXTRAL 2-RANK chain F — BASELINE FAIRNESS PASS (Kevin's challenge: all-dead
# baselines look odd). Every chain-A/B baseline death was the watchdog firing
# at 49 GiB vs its default 50-GiB floor DURING LOAD — marginal kills. House
# precedent (llama/q3.5 sEP campaigns) runs capacity work at floor 35-40.
# Rerun the baselines at HOST_MEM_WATCHDOG_FLOOR_GB=35; any fit rebuilds the
# panel with real turning points.
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
export GPU_POOL=0,1
export HOST_MEM_WATCHDOG_FLOOR_GB=35
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
  echo "=== $tag @$seq floor35 b-walk: $* $(date -u +%H:%M:%S)"
  MAX_SAMPLES=512 bash "$P" mixtral-8x22b "$tag" "$cfg" "$seq" "$@"
  echo "MX2R_${tag}_EXIT=$?"
}
set +e
run mx2frc32 "superoffload_mem|recomp|ligerloss1"      32000 4 2 1
run mx2fun32 "superoffload_mem|unsloth|ligerloss1"     32000 4 2 1
run mx2fuo32 "superoffload_mem|unsloth-off|ligerloss1" 32000 2 1
run mx2fz332 "zero3_offload_mem|recomp|ligerloss1"     32000 2 1
run mx2frc64 "superoffload_mem|recomp|ligerloss1"      64000 2 1
run mx2fun64 "superoffload_mem|unsloth|ligerloss1"     64000 2 1
echo "=== MX2R CHAIN-F DONE $(date -u +%H:%M:%S)"
