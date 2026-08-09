#!/bin/bash
# FA4 remeasure ladders: q3.5-35b + glm4.7-flash NeMo walls with FlashAttention
# engaged (.venv-nemo-fa4 via driver routing). 2r EP2 first, then 1r. Climb to
# wall; also confirm the 2r panel rung (256k) explicitly for both.
set -uo pipefail
export XDG_CACHE_HOME=/scratch_local/user_data/shutian/kevin/cache
cd /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM
S="agent/anchors_tmp/nemo_fa4_probe_status.log"
cell() { # $1 model $2 ranks $3 seq
  local gpus="0"; [ "$2" = "2" ] && gpus="0,1"
  local tag="${1//./_}_r$2_s$(( $3 / 1000 ))"
  echo "START fa4lad $tag $(date +%H:%M:%S)" >> "$S"
  NEMO_DEBUG_BATCH=1 \
  RUNS="$1|$2 ; nemo|recomp|ligerloss1 ; $3|1|1 ; none|false|false|false|false|false" \
  RUN_NAME=nemofa4 GPU_POOL="$gpus" RUN_TIMEOUT_SECONDS=3600 OVERWRITE=false \
    bash scripts/lf/profile_lora_nemo.sh >> "agent/anchors_tmp/nemofa4_lad_${tag}.log" 2>&1
  v=$(grep -o 'VERDICT=[A-Z]*' "agent/anchors_tmp/nemofa4_lad_${tag}.log" | tail -1)
  echo "CELL fa4lad $tag -> ${v:-NONE} $(date +%H:%M:%S)" >> "$S"
  echo "${v#VERDICT=}"
}
for m in q3.5-35b-a3b glm4.7-flash; do
  for s in 64000 96000 128000 192000 256000 320000; do
    v=$(cell "$m" 2 "$s")
    [ "$v" = "TRAINED" ] || { echo "FA4 $m r2 wall at $s ($v)" >> "$S"; break; }
  done
done
for m in q3.5-35b-a3b glm4.7-flash; do
  for s in 32000 64000 96000 128000 192000; do
    v=$(cell "$m" 1 "$s")
    [ "$v" = "TRAINED" ] || { echo "FA4 $m r1 wall at $s ($v)" >> "$S"; break; }
  done
done
echo "FA4LADDERS_DONE $(date +%H:%M:%S)" >> "$S"
