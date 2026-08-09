#!/bin/bash
# NeMo baseline campaign — q3-30b-a3b, 2 ranks, EP2, serial (project rules).
# Phase 1: the tp2r plot rung 384k (expect GOOM) on both arms.
# Phase 2: wall hunt from below — climb until first GOOM per arm.
# All runs w1+m2, b1 ga1, LoRA r64/a16/drop0.
set -uo pipefail
export XDG_CACHE_HOME=/scratch_local/user_data/shutian/kevin/cache
cd /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM
S="agent/anchors_tmp/nemo_q30b_status.log"
run_cell() { # $1 tag  $2 recompute  $3 seq  $4 batch
  echo "START $1 $2 s=$3 b=$4 $(date +%H:%M:%S)" >> "$S"
  RUNS="q3-30b-a3b|2 ; nemo|$2|ligerloss1 ; $3|$4|1 ; none|false|false|false|false|false" \
  RUN_NAME=nemo30b GPU_POOL=0,1 RUN_TIMEOUT_SECONDS=${RUN_TIMEOUT_SECONDS:-5400} OVERWRITE=${OVERWRITE:-false} \
    bash scripts/lf/profile_lora_nemo.sh >> "agent/anchors_tmp/nemo30b_${1}.log" 2>&1
  v=$(grep -o 'VERDICT=[A-Z]*' "agent/anchors_tmp/nemo30b_${1}.log" | tail -1)
  echo "CELL $1 $2 s=$3 b=$4 -> ${v:-NONE} $(date +%H:%M:%S)" >> "$S"
  echo "${v#VERDICT=}"
}

# ── Phase 1: plot rung 384k, both arms (expected GOOM) ──
run_cell rc384 recomp 384000 1
run_cell ao384 actoff 384000 1

# ── Phase 2: wall hunt, recomp arm: climb 32k -> 64k -> 96k -> 128k -> 160k -> 192k -> 256k ──
for s in 32000 64000 96000 128000 160000 192000 256000; do
  v=$(run_cell "rc$((s/1000))" recomp "$s" 1)
  [ "$v" = "GOOM" ] && { echo "recomp wall at $s" >> "$S"; break; }
  [ "$v" = "COOM" ] && { echo "recomp host wall at $s" >> "$S"; break; }
  [ "$v" = "FAIL" ] && { echo "recomp FAIL at $s (inspect)" >> "$S"; break; }
done

# ── Phase 2b: wall hunt, actoff arm ──
for s in 32000 64000 96000 128000 160000 192000 256000; do
  v=$(run_cell "ao$((s/1000))" actoff "$s" 1)
  [ "$v" = "GOOM" ] && { echo "actoff wall at $s" >> "$S"; break; }
  [ "$v" = "COOM" ] && { echo "actoff host wall at $s" >> "$S"; break; }
  [ "$v" = "FAIL" ] && { echo "actoff FAIL at $s (inspect)" >> "$S"; break; }
done

echo "CHAIN_DONE $(date +%H:%M:%S)" >> "$S"
echo CHAIN_DONE
