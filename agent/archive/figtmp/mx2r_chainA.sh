#!/bin/bash
# MIXTRAL 2-RANK tp panel campaign, chain A (INSIDE container): rungs 32k+64k.
# Per GLMTP_CAMPAIGN.md mechanics: RUNS model|2, GPU_POOL=0,1, DDP_TIMEOUT
# 7200 (driver default), asym 2r backend = asym_sdp2_cpuadamwds (arms
# ASYM_ARENA_SHM=1 itself; cap raised to 300 for the 271-GB mixtral bank),
# w1+m2, MAX_SAMPLES=512, serial+solo, best-batch = walk-down first-fit
# (tp2_probe tries batches in order). Cells = GLOBAL tok/s = 2x per-rank eff.
# Per rung order: rc -> un -> uo -> asym(T1). Tags mx2<sys><seq>.
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

run() { # $1 tag-prefix  $2 config  $3 seq  $4... batches
  local tag="$1" cfg="$2" seq="$3"; shift 3
  guard "$tag"
  echo "=== $tag @$seq b-walk: $* $(date -u +%H:%M:%S)"
  MAX_SAMPLES=512 bash "$P" mixtral-8x22b "$tag" "$cfg" "$seq" "$@"
  echo "MX2R_${tag}_EXIT=$?"
}

# ── 32k rung ──
run mx2rc32 "superoffload_mem|recomp|ligerloss1"      32000 8 6 4
run mx2un32 "superoffload_mem|unsloth|ligerloss1"     32000 8 6 4
run mx2uo32 "superoffload_mem|unsloth-off|ligerloss1" 32000 4 2 1
run mx2t132 "asym_sdp2_cpuadamwds|T1|ligerloss1"      32000 8 6 4

# ── 64k rung ──
run mx2rc64 "superoffload_mem|recomp|ligerloss1"      64000 4 3 2
run mx2un64 "superoffload_mem|unsloth|ligerloss1"     64000 4 3 2
run mx2uo64 "superoffload_mem|unsloth-off|ligerloss1" 64000 2 1
run mx2t164 "asym_sdp2_cpuadamwds|T1|ligerloss1"      64000 4 3 2

echo "=== MX2R CHAIN-A DONE $(date -u +%H:%M:%S)"
