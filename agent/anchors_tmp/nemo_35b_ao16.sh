#!/bin/bash
set -uo pipefail
export XDG_CACHE_HOME=/scratch_local/user_data/shutian/kevin/cache
cd /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM
S2="agent/anchors_tmp/nemo_q35b_status.log"
echo "START ao16 s=16000 $(date +%H:%M:%S)" >> "$S2"
RUNS='q3.5-35b-a3b|2 ; nemo|actoff|ligerloss1 ; 16000|1|1 ; none|false|false|false|false|false' \
RUN_NAME=nemo35b GPU_POOL=0,1 RUN_TIMEOUT_SECONDS=3600 OVERWRITE=false \
  bash scripts/lf/profile_lora_nemo.sh >> agent/anchors_tmp/nemo35b_ao16.log 2>&1
v=$(grep -o 'VERDICT=[A-Z]*' agent/anchors_tmp/nemo35b_ao16.log | tail -1)
echo "CELL ao16 s=16000 -> ${v:-NONE} $(date +%H:%M:%S)" >> "$S2"
echo AO16_DONE >> "$S2"
