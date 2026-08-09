#!/bin/bash
set -uo pipefail
export XDG_CACHE_HOME=/scratch_local/user_data/shutian/kevin/cache
cd /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM
S="agent/anchors_tmp/nemo_1rank_status.log"
export NEMO_DEBUG_BATCH=1
for r in 1 2; do
  echo "START diag35b-r$r s=16000 $(date +%H:%M:%S)" >> "$S"
  RUNS="q3.5-35b-a3b|$r ; nemo|recomp|ligerloss1 ; 16000|1|1 ; none|false|false|false|false|false" \
  RUN_NAME=diag35b GPU_POOL=$([ "$r" = 1 ] && echo 0 || echo 0,1) RUN_TIMEOUT_SECONDS=3600 OVERWRITE=true \
    bash scripts/lf/profile_lora_nemo.sh >> "agent/anchors_tmp/nemo35b_diag_r${r}.log" 2>&1
  v=$(grep -o 'VERDICT=[A-Z]*' "agent/anchors_tmp/nemo35b_diag_r${r}.log" | tail -1)
  echo "CELL diag35b-r$r -> ${v:-NONE} $(date +%H:%M:%S)" >> "$S"
done
echo DIAG35_DONE >> "$S"
