#!/bin/bash
# Download the two nemo-campaign checkpoints into the shared HF cache (idempotent).
set -uo pipefail
export HF_HOME=/scratch_local/user_data/shutian/kevin/cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
PY=/workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/.venv/bin/python
for repo in Qwen/Qwen3-30B-A3B Qwen/Qwen3.5-35B-A3B; do
  echo "=== downloading ${repo} $(date +%H:%M:%S)"
  "${PY}" - "$repo" <<'EOF'
import sys
from huggingface_hub import snapshot_download
repo = sys.argv[1]
p = snapshot_download(repo, max_workers=8)
print("DONE", repo, p)
EOF
  rc=$?
  echo "=== ${repo} rc=${rc} $(date +%H:%M:%S)"
  [ $rc -ne 0 ] && exit $rc
done
echo ALL_DOWNLOADS_DONE
