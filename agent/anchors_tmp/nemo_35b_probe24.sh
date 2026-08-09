#!/bin/bash
set -uo pipefail
export XDG_CACHE_HOME=/scratch_local/user_data/shutian/kevin/cache
cd /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM
S2="agent/anchors_tmp/nemo_q35b_status.log"
for arm in recomp actoff; do
  echo "START ${arm}24 s=24000 $(date +%H:%M:%S)" >> "$S2"
  RUNS="q3.5-35b-a3b|2 ; nemo|${arm}|ligerloss1 ; 24000|1|1 ; none|false|false|false|false|false" \
  RUN_NAME=nemo35b GPU_POOL=0,1 RUN_TIMEOUT_SECONDS=3600 OVERWRITE=false \
    bash scripts/lf/profile_lora_nemo.sh >> "agent/anchors_tmp/nemo35b_${arm}24.log" 2>&1
  v=$(grep -o 'VERDICT=[A-Z]*' "agent/anchors_tmp/nemo35b_${arm}24.log" | tail -1)
  echo "CELL ${arm}24 s=24000 -> ${v:-NONE} $(date +%H:%M:%S)" >> "$S2"
done
echo PROBE24_DONE >> "$S2"
echo PROBE24_DONE
