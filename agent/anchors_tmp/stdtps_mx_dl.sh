#!/bin/bash
# stdtps_mx_dl.sh — Agent-2 prerequisite: download mistralai/Mixtral-8x22B-v0.1
# into this box's HF cache (disk/network only, no GPU). The fused-format local
# copy (2r load requirement) is built AFTERWARD, between GPU cells, with
# agent/anchors_tmp/mx_fuse_local.py.
cd /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM
export HF_HOME=/scratch_local/user_data/shutian/kevin/cache/huggingface
S=/workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/stdtps_status.log
echo "STDTPS-MX-DL BEGIN $(date '+%F %H:%M:%S')" >> "$S"
.venv/bin/python - <<'EOF'
from huggingface_hub import snapshot_download
p = snapshot_download("mistralai/Mixtral-8x22B-v0.1", max_workers=8)
print("DL-DONE", p, flush=True)
EOF
rc=$?
echo "STDTPS-MX-DL rc=$rc $(date '+%F %H:%M:%S')" >> "$S"
