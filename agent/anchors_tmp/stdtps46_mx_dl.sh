#!/bin/bash
# stdtps46_mx_dl.sh — Agent-2 prerequisite on c18: download
# mistralai/Mixtral-8x22B-v0.1 into this box's HF cache (network only, no
# GPU), then build the FUSED local copy the driver M-map points at
# (mx_fuse_local.py, CPU/disk only). Both steps idempotent/resumable.
cd /workspace/AsymGEMM-SFT-46/third_party/AsymGEMM
export HF_HOME=/scratch_local/user_data/shutian/kevin/cache/huggingface
S=/workspace/AsymGEMM-SFT-46/third_party/AsymGEMM/agent/anchors_tmp/stdtps46_status.log
echo "MX-DL BEGIN $(date '+%F %H:%M:%S')" >> "$S"
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("mistralai/Mixtral-8x22B-v0.1", max_workers=8,
                      allow_patterns=["*.safetensors*", "*.json", "tokenizer*", "*.model"])
print("DL-DONE", p, flush=True)
PY
rc=$?
echo "MX-DL rc=$rc $(date '+%F %H:%M:%S')" >> "$S"
[ $rc -ne 0 ] && exit $rc
dst=/scratch_local/user_data/shutian/kevin/cache/fused/Mixtral-8x22B-v0.1
if [ -f "$dst/model.safetensors.index.json" ] && [ -f "$dst/config.json" ]; then
  echo "MX-FUSE already present, skip $(date '+%F %H:%M:%S')" >> "$S"; exit 0
fi
echo "MX-FUSE BEGIN $(date '+%F %H:%M:%S')" >> "$S"
.venv/bin/python agent/anchors_tmp/mx_fuse_local.py
rc=$?
echo "MX-FUSE rc=$rc $(date '+%F %H:%M:%S')" >> "$S"
