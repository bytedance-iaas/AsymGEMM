#!/bin/bash
# 35b 1-rank ladder v2 — with the forced pad_to_max_length collate guard.
# NEMO_DEBUG_BATCH=1 so every train.log carries the actual batch shapes.
set -uo pipefail
export XDG_CACHE_HOME=/scratch_local/user_data/shutian/kevin/cache
cd /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM
S="agent/anchors_tmp/nemo_1rank_status.log"
export NEMO_DEBUG_BATCH=1
run_cell() { # $1 tag $2 arm $3 seq
  echo "START $1 $2 s=$3 $(date +%H:%M:%S)" >> "$S"
  RUNS="q3.5-35b-a3b|1 ; nemo|$2|ligerloss1 ; $3|1|1 ; none|false|false|false|false|false" \
  RUN_NAME=nemo1r GPU_POOL=0 RUN_TIMEOUT_SECONDS=3600 OVERWRITE=true \
    bash scripts/lf/profile_lora_nemo.sh >> "agent/anchors_tmp/nemo1r_${1}.log" 2>&1
  v=$(grep -o 'VERDICT=[A-Z]*' "agent/anchors_tmp/nemo1r_${1}.log" | tail -1)
  echo "CELL $1 q3.5-35b-a3b $2 s=$3 -> ${v:-NONE} $(date +%H:%M:%S)" >> "$S"
  echo "${v#VERDICT=}"
}
for s in 8000 16000 24000 32000; do
  v=$(run_cell "v2rc$((s/1000))" recomp "$s")
  [ "$v" = "TRAINED" ] || { echo "35b recomp 1r wall at $s ($v)" >> "$S"; break; }
done
run_cell v2rc128 recomp 128000
run_cell v2ao16 actoff 16000
echo V2_35B_DONE >> "$S"
