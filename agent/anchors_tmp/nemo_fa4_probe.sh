#!/bin/bash
# FA4 probe pair: do qwen3.5-35b + glm4.7-flash NeMo cells escape the unfused
# O(S^2) fallback when run in .venv-nemo-fa4 (TE v4_is_installed=True)?
# Same rungs that GOOMed on the base venv (EP2 @32k recomp).
set -uo pipefail
export XDG_CACHE_HOME=/scratch_local/user_data/shutian/kevin/cache
cd /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM
S="agent/anchors_tmp/nemo_fa4_probe_status.log"
for m in q3.5-35b-a3b glm4.7-flash; do
  echo "START fa4probe $m s=32000 $(date +%H:%M:%S)" >> "$S"
  NEMO_DEBUG_BATCH=1 \
  RUNS="$m|2 ; nemo|recomp|ligerloss1 ; 32000|1|1 ; none|false|false|false|false|false" \
  RUN_NAME=nemofa4 GPU_POOL=0,1 RUN_TIMEOUT_SECONDS=3600 OVERWRITE=false \
    bash scripts/lf/profile_lora_nemo.sh >> "agent/anchors_tmp/nemofa4_${m//./_}.log" 2>&1
  v=$(grep -o 'VERDICT=[A-Z]*' "agent/anchors_tmp/nemofa4_${m//./_}.log" | tail -1)
  echo "CELL fa4probe $m s=32000 -> ${v:-NONE} $(date +%H:%M:%S)" >> "$S"
done
echo "FA4PROBE_DONE $(date +%H:%M:%S)" >> "$S"
