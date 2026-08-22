#!/bin/bash
cd /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM
export HF_HOME=/scratch_local/user_data/shutian/kevin/cache/huggingface
S=agent/anchors_tmp/stdtps96_status.log
echo "STDTPS96-G20-DEQUANT BEGIN $(date '+%F %H:%M:%S')" >> "$S"
.venv/bin/python agent/anchors_tmp/stdtps96_g20_dequant.py
rc=$?
echo "STDTPS96-G20-DEQUANT rc=$rc $(date '+%F %H:%M:%S')" >> "$S"
