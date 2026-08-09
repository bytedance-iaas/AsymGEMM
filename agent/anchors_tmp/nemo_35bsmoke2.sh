#!/bin/bash
set -uo pipefail
export XDG_CACHE_HOME=/scratch_local/user_data/shutian/kevin/cache
cd /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM
S2="agent/anchors_tmp/nemo_q35b_status.log"
echo "START smoke35b-v2 recomp s=16000 $(date +%H:%M:%S)" >> "$S2"
RUNS='q3.5-35b-a3b|2 ; nemo|recomp|ligerloss1 ; 16000|1|1 ; none|false|false|false|false|false' \
RUN_NAME=smoke35b GPU_POOL=0,1 RUN_TIMEOUT_SECONDS=3600 OVERWRITE=true \
  bash scripts/lf/profile_lora_nemo.sh >> agent/anchors_tmp/nemo35b_smoke2.log 2>&1
v=$(grep -o 'VERDICT=[A-Z]*' agent/anchors_tmp/nemo35b_smoke2.log | tail -1)
echo "CELL smoke35b-v2 -> ${v:-NONE} $(date +%H:%M:%S)" >> "$S2"
echo BATCH_DONE
