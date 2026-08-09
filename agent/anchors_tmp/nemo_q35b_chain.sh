#!/bin/bash
# NeMo baseline campaign — q3.5-35b-a3b (VL checkpoint, text-only), 2 ranks EP2.
# Phase 1: tp2r plot rung 256k (expect GOOM) both arms; Phase 2: wall hunt.
set -uo pipefail
export XDG_CACHE_HOME=/scratch_local/user_data/shutian/kevin/cache
cd /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM
S="agent/anchors_tmp/nemo_q35b_status.log"
run_cell() { # $1 tag  $2 recompute  $3 seq  $4 batch
  echo "START $1 $2 s=$3 b=$4 $(date +%H:%M:%S)" >> "$S"
  RUNS="q3.5-35b-a3b|2 ; nemo|$2|ligerloss1 ; $3|$4|1 ; none|false|false|false|false|false" \
  RUN_NAME=nemo35b GPU_POOL=0,1 RUN_TIMEOUT_SECONDS=${RUN_TIMEOUT_SECONDS:-5400} OVERWRITE=${OVERWRITE:-false} \
    bash scripts/lf/profile_lora_nemo.sh >> "agent/anchors_tmp/nemo35b_${1}.log" 2>&1
  v=$(grep -o 'VERDICT=[A-Z]*' "agent/anchors_tmp/nemo35b_${1}.log" | tail -1)
  echo "CELL $1 $2 s=$3 b=$4 -> ${v:-NONE} $(date +%H:%M:%S)" >> "$S"
  echo "${v#VERDICT=}"
}

# ── Phase 1: plot rung 256k, both arms (expected GOOM) ──
run_cell rc256 recomp 256000 1
run_cell ao256 actoff 256000 1

# ── Phase 2: wall hunt, recomp arm ──
for s in 32000 64000 96000 128000 160000 192000; do
  v=$(run_cell "rc$((s/1000))" recomp "$s" 1)
  [ "$v" = "GOOM" ] && { echo "recomp wall at $s" >> "$S"; break; }
  [ "$v" = "COOM" ] && { echo "recomp host wall at $s" >> "$S"; break; }
  [ "$v" = "FAIL" ] && { echo "recomp FAIL at $s (inspect)" >> "$S"; break; }
done

# ── Phase 2b: wall hunt, actoff arm ──
for s in 32000 64000 96000 128000 160000 192000; do
  v=$(run_cell "ao$((s/1000))" actoff "$s" 1)
  [ "$v" = "GOOM" ] && { echo "actoff wall at $s" >> "$S"; break; }
  [ "$v" = "COOM" ] && { echo "actoff host wall at $s" >> "$S"; break; }
  [ "$v" = "FAIL" ] && { echo "actoff FAIL at $s (inspect)" >> "$S"; break; }
done

echo "CHAIN_DONE $(date +%H:%M:%S)" >> "$S"
echo CHAIN_DONE
