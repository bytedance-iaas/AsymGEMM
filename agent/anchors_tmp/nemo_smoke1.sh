#!/bin/bash
# In-container smoke: q3-30b-a3b EP2 LoRA, seq 8192, recomp arm.
set -uo pipefail
export XDG_CACHE_HOME=/scratch_local/user_data/shutian/kevin/cache
cd /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM
RUNS='q3-30b-a3b|2 ; nemo|recomp|ligerloss1 ; 8192|1|1 ; none|false|false|false|false|false' \
RUN_NAME=smoke1 GPU_POOL=0,1 RUN_TIMEOUT_SECONDS=3600 \
bash scripts/lf/profile_lora_nemo.sh
echo "SMOKE_RC=$?"
