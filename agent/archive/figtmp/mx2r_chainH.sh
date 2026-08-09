#!/bin/bash
# MIXTRAL 2-RANK chain H — STAGGERED-LOAD baseline probe. Root cause found
# (Kevin's challenge, round 2): transformers' zero3 loader runs the mixtral
# expert-stack weight conversion on CPU in EVERY rank pre-partitioning ->
# ~full-model transient per rank (580 GB measured) -> concurrent loads blow
# the 957-GB node. ASYM_STAGGER_RANK_LOAD=1 (new harness flag) serializes
# rank loads. If baselines now stand, the panel gets its true cascade.
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
export GPU_POOL=0,1
export HOST_MEM_WATCHDOG_FLOOR_GB=35
export ASYM_STAGGER_RANK_LOAD=1
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
  echo "=== $tag @$seq stagger+floor35 b-walk: $* $(date -u +%H:%M:%S)"
  MAX_SAMPLES=512 bash "$P" mixtral-8x22b "$tag" "$cfg" "$seq" "$@"
  echo "MX2R_${tag}_EXIT=$?"
}
set +e
run mx2src32 "superoffload_mem|recomp|ligerloss1"      32000 4 2 1
run mx2sun32 "superoffload_mem|unsloth|ligerloss1"     32000 4 2 1
run mx2suo32 "superoffload_mem|unsloth-off|ligerloss1" 32000 2 1
run mx2sz332 "zero3_offload_mem|recomp|ligerloss1"     32000 2 1
run mx2src64 "superoffload_mem|recomp|ligerloss1"      64000 2 1
run mx2sun64 "superoffload_mem|unsloth|ligerloss1"     64000 2 1
echo "=== MX2R CHAIN-H DONE $(date -u +%H:%M:%S)"
