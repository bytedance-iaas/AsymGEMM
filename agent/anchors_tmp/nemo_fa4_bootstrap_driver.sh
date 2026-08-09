#!/bin/bash
# In-container driver for bootstrap_nemo_venv_fa4.sh (fresh build — also serves
# as the clean-reproducibility validation of bootstrap_nemo_venv.sh).
set -uo pipefail
export XDG_CACHE_HOME=/scratch_local/user_data/shutian/kevin/cache
export HF_HOME=/scratch_local/user_data/shutian/kevin/cache/huggingface
cd /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM
RECREATE_ENV=1 bash scripts/lf/bootstrap_nemo_venv_fa4.sh
rc=$?
echo "FA4_BOOTSTRAP_RC=${rc}"
exit $rc
