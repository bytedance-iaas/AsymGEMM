#!/bin/bash
# In-container replica wrapper for the EP skew probe (one model, N datasets).
# Usage: run_cell.sh MODEL_KEY DS1[,DS2,...] [EXTRA_ARGS...]
set -uo pipefail
MODEL=$1; DATASETS=$2; shift 2 || true
export HF_HOME=${HF_HOME:-/scratch_local/user_data/shutian/kevin/cache/huggingface}
if [ -z "${HF_TOKEN:-}" ] && [ -f /workspace/env/bashrc.sh ]; then
  export HF_TOKEN=$(grep -m1 'export HF_TOKEN=' /workspace/env/bashrc.sh | cut -d'"' -f2)
fi
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
exec .venv/bin/python scripts/ep_skew/route_skew_probe.py \
  --model "$MODEL" --datasets "$DATASETS" "$@"
