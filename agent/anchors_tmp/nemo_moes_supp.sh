#!/bin/bash
# Supplemental: actoff cells at fitting seqs + glm4.5 32k NCCL@edge re-probe.
set -uo pipefail
export XDG_CACHE_HOME=/scratch_local/user_data/shutian/kevin/cache
export HF_HOME=/scratch_local/user_data/shutian/kevin/cache/huggingface
cd /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM
S="agent/anchors_tmp/nemo_moes_status.log"
run_cell() { # $1 tag $2 model $3 ranks $4 arm $5 seq $6 overwrite
  local gpus="0"; [ "$3" = "2" ] && gpus="0,1"
  echo "START $1 $2 r$3 $4 s=$5 $(date +%H:%M:%S)" >> "$S"
  NEMO_DEBUG_BATCH=1 \
  RUNS="$2|$3 ; nemo|$4|ligerloss1 ; $5|1|1 ; none|false|false|false|false|false" \
  RUN_NAME=nemomoe GPU_POOL="$gpus" RUN_TIMEOUT_SECONDS=5400 OVERWRITE="${6:-false}" \
    bash scripts/lf/profile_lora_nemo.sh >> "agent/anchors_tmp/nemomoe_${1}.log" 2>&1
  v=$(grep -o 'VERDICT=[A-Z]*' "agent/anchors_tmp/nemomoe_${1}.log" | tail -1)
  echo "CELL $1 $2 r$3 $4 s=$5 -> ${v:-NONE} $(date +%H:%M:%S)" >> "$S"
}
run_cell g47r2ao16 glm4.7-flash 2 actoff 16000
run_cell g45r2ao8 glm4.5-air 2 actoff 8000
run_cell g45r2rc32b glm4.5-air 2 recomp 32000 true
echo SUPP_DONE >> "$S"
