#!/bin/bash
# NeMo baseline — 1-RANK (single GPU, EP1: full model resident) ladders.
# 30b recomp: 32k..128k incl. the 80k plot rung; 35b recomp: 8k..24k;
# one actoff spot cell on 30b@32k. Serial, GPU 0.
set -uo pipefail
export XDG_CACHE_HOME=/scratch_local/user_data/shutian/kevin/cache
cd /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM
S="agent/anchors_tmp/nemo_1rank_status.log"
run_cell() { # $1 tag $2 model $3 arm $4 seq
  echo "START $1 $2 $3 s=$4 $(date +%H:%M:%S)" >> "$S"
  RUNS="$2|1 ; nemo|$3|ligerloss1 ; $4|1|1 ; none|false|false|false|false|false" \
  RUN_NAME=nemo1r GPU_POOL=0 RUN_TIMEOUT_SECONDS=5400 OVERWRITE=false \
    bash scripts/lf/profile_lora_nemo.sh >> "agent/anchors_tmp/nemo1r_${1}.log" 2>&1
  v=$(grep -o 'VERDICT=[A-Z]*' "agent/anchors_tmp/nemo1r_${1}.log" | tail -1)
  echo "CELL $1 $2 $3 s=$4 -> ${v:-NONE} $(date +%H:%M:%S)" >> "$S"
  echo "${v#VERDICT=}"
}

for s in 32000 64000 80000 96000 128000; do
  v=$(run_cell "30rc$((s/1000))" q3-30b-a3b recomp "$s")
  [ "$v" = "TRAINED" ] || { echo "30b recomp 1r wall at $s ($v)" >> "$S"; break; }
done

run_cell 30ao32 q3-30b-a3b actoff 32000

for s in 8000 16000 24000; do
  v=$(run_cell "35rc$((s/1000))" q3.5-35b-a3b recomp "$s")
  [ "$v" = "TRAINED" ] || { echo "35b recomp 1r wall at $s ($v)" >> "$S"; break; }
done

echo CHAIN1R_DONE >> "$S"
