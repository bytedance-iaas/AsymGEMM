#!/bin/bash
# Mixtral NeMo chain: 2r descent 8k->4k, actoff spot at first fit, 1r load probe.
set -uo pipefail
export XDG_CACHE_HOME=/scratch_local/user_data/shutian/kevin/cache
cd /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM
S="agent/anchors_tmp/nemo_moes_status.log"
run_cell() { # $1 tag $2 ranks $3 arm $4 seq
  local gpus="0"; [ "$2" = "2" ] && gpus="0,1"
  echo "START $1 mixtral r$2 $3 s=$4 $(date +%H:%M:%S)" >> "$S"
  NEMO_DEBUG_BATCH=1 \
  RUNS="mixtral-8x22b|$2 ; nemo|$3|ligerloss1 ; $4|1|1 ; none|false|false|false|false|false" \
  RUN_NAME=nemomoe GPU_POOL="$gpus" RUN_TIMEOUT_SECONDS=5400 OVERWRITE=false \
    bash scripts/lf/profile_lora_nemo.sh >> "agent/anchors_tmp/nemomoe_${1}.log" 2>&1
  v=$(grep -o 'VERDICT=[A-Z]*' "agent/anchors_tmp/nemomoe_${1}.log" | tail -1)
  echo "CELL $1 mixtral r$2 $3 s=$4 -> ${v:-NONE} $(date +%H:%M:%S)" >> "$S"
  echo "${v#VERDICT=}"
}
fit=""
for s in 8000 4000; do
  v=$(run_cell "mxr2rc$((s/1000))" 2 recomp "$s")
  [ "$v" = "TRAINED" ] && { fit=$s; break; }
done
if [ -n "$fit" ]; then
  run_cell "mxr2ao$((fit/1000))" 2 actoff "$fit"
else
  echo "mixtral r2 recomp: no fit >=4k" >> "$S"
fi
run_cell mxr1load 1 recomp 4000
echo MX_DONE >> "$S"
