#!/bin/bash
# In-container driver for bootstrap_nemo_venv.sh (non-interactive).
set -uo pipefail
export XDG_CACHE_HOME=/scratch_local/user_data/shutian/kevin/cache
export HF_HOME=/scratch_local/user_data/shutian/kevin/cache/huggingface
cd /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM
bash scripts/lf/bootstrap_nemo_venv.sh
rc=$?
echo "BOOTSTRAP_RC=${rc}"
exit $rc
