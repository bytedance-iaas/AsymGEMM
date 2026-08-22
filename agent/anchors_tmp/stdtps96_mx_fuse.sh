#!/bin/bash
# stdtps96: build the FUSED Mixtral local checkpoint on THIS node (2r load req).
cd /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM
export HF_HOME=/scratch_local/user_data/shutian/kevin/cache/huggingface
S=agent/anchors_tmp/stdtps96_status.log
echo "STDTPS96-MX-FUSE BEGIN $(date '+%F %H:%M:%S')" >> "$S"
.venv/bin/python agent/anchors_tmp/mx_fuse_local.py
rc=$?
ls -la /scratch_local/user_data/shutian/kevin/cache/fused/Mixtral-8x22B-v0.1/ 2>/dev/null | head -3 >> "$S"
echo "STDTPS96-MX-FUSE rc=$rc $(date '+%F %H:%M:%S')" >> "$S"
