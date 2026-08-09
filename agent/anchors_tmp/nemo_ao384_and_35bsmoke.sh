#!/bin/bash
# Serial: (1) re-run q30b ao384 with NVTE_CPU_OFFLOAD_V1 fix; (2) q3.5-35b smoke 16k.
set -uo pipefail
export XDG_CACHE_HOME=/scratch_local/user_data/shutian/kevin/cache
cd /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM
S="agent/anchors_tmp/nemo_q30b_status.log"

echo "START ao384-redo $(date +%H:%M:%S)" >> "$S"
RUNS='q3-30b-a3b|2 ; nemo|actoff|ligerloss1 ; 384000|1|1 ; none|false|false|false|false|false' \
RUN_NAME=nemo30b GPU_POOL=0,1 RUN_TIMEOUT_SECONDS=5400 OVERWRITE=true \
  bash scripts/lf/profile_lora_nemo.sh >> agent/anchors_tmp/nemo30b_ao384redo.log 2>&1
v=$(grep -o 'VERDICT=[A-Z]*' agent/anchors_tmp/nemo30b_ao384redo.log | tail -1)
echo "CELL ao384-redo -> ${v:-NONE} $(date +%H:%M:%S)" >> "$S"

S2="agent/anchors_tmp/nemo_q35b_status.log"
echo "START smoke35b recomp s=16000 $(date +%H:%M:%S)" >> "$S2"
RUNS='q3.5-35b-a3b|2 ; nemo|recomp|ligerloss1 ; 16000|1|1 ; none|false|false|false|false|false' \
RUN_NAME=smoke35b GPU_POOL=0,1 RUN_TIMEOUT_SECONDS=3600 OVERWRITE=true \
  bash scripts/lf/profile_lora_nemo.sh >> agent/anchors_tmp/nemo35b_smoke.log 2>&1
v=$(grep -o 'VERDICT=[A-Z]*' agent/anchors_tmp/nemo35b_smoke.log | tail -1)
echo "CELL smoke35b -> ${v:-NONE} $(date +%H:%M:%S)" >> "$S2"
echo BATCH_DONE
