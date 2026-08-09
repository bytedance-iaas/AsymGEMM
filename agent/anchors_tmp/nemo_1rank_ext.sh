#!/bin/bash
# 35b 1-rank extension: climb 32k/48k/64k to the wall, measured 128k rung, ao16.
set -uo pipefail
export XDG_CACHE_HOME=/scratch_local/user_data/shutian/kevin/cache
cd /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM
S="agent/anchors_tmp/nemo_1rank_status.log"
run_cell() { # $1 tag $2 arm $3 seq
  echo "START $1 $2 s=$3 $(date +%H:%M:%S)" >> "$S"
  RUNS="q3.5-35b-a3b|1 ; nemo|$2|ligerloss1 ; $3|1|1 ; none|false|false|false|false|false" \
  RUN_NAME=nemo1r GPU_POOL=0 RUN_TIMEOUT_SECONDS=5400 OVERWRITE=false \
    bash scripts/lf/profile_lora_nemo.sh >> "agent/anchors_tmp/nemo1r_${1}.log" 2>&1
  v=$(grep -o 'VERDICT=[A-Z]*' "agent/anchors_tmp/nemo1r_${1}.log" | tail -1)
  echo "CELL $1 q3.5-35b-a3b $2 s=$3 -> ${v:-NONE} $(date +%H:%M:%S)" >> "$S"
  echo "${v#VERDICT=}"
}
for s in 32000 48000 64000; do
  v=$(run_cell "35rc$((s/1000))" recomp "$s")
  [ "$v" = "TRAINED" ] || { echo "35b recomp 1r wall at $s ($v)" >> "$S"; break; }
done
run_cell 35rc128 recomp 128000
run_cell 35ao16 actoff 16000
echo EXT1R_DONE >> "$S"
