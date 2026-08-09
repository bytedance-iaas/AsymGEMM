#!/bin/bash
# MIXTRAL 2-RANK chain I — baselines from the FUSED checkpoint (no load-time
# expert conversion). V = 1-rank validation probe first (fused load + train
# sanity), then the 2-rank ladder at floor 35 with shm hygiene.
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
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
set +e

guard V1
echo "=== V: 1-rank rc @32k b1 fused-load validation (w1+m1) $(date -u +%H:%M:%S)"
GPU_POOL=0 WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 \
  bash scripts/lf/tp_probe.sh mixtral-8x22b mxfusev1 "superoffload_mem|recomp|ligerloss1" 32000 1
V=$?
echo "MX2R_V_EXIT=$V"
if [ $V -ne 0 ]; then echo "CHAINI_STATE=validation_failed"; exit 3; fi

export GPU_POOL=0,1
P=".figtmp/tp2_probe.sh"
run() {
  local tag="$1" cfg="$2" seq="$3"; shift 3
  guard "$tag"
  echo "=== $tag @$seq fused+floor35 b-walk: $* $(date -u +%H:%M:%S)"
  MAX_SAMPLES=512 bash "$P" mixtral-8x22b "$tag" "$cfg" "$seq" "$@"
  echo "MX2R_${tag}_EXIT=$?"
}
run mx2Frc32 "superoffload_mem|recomp|ligerloss1"      32000 4 2 1
run mx2Fun32 "superoffload_mem|unsloth|ligerloss1"     32000 4 2 1
run mx2Fuo32 "superoffload_mem|unsloth-off|ligerloss1" 32000 2 1
run mx2Fz332 "zero3_offload_mem|recomp|ligerloss1"     32000 2 1
run mx2Frc64 "superoffload_mem|recomp|ligerloss1"      64000 2 1
run mx2Fun64 "superoffload_mem|unsloth|ligerloss1"     64000 2 1
echo "=== MX2R CHAIN-I DONE $(date -u +%H:%M:%S)"
