#!/bin/bash
# stdtps96_g20_build.sh — Agent-4 prerequisite on c18: download gpt-oss-20b and
# build the DEQUANTIZED bf16 fused local copy (CPU/network only, no GPU).
cd /workspace/AsymGEMM-SFT-46/third_party/AsymGEMM
export HF_HOME=/scratch_local/user_data/shutian/kevin/cache/huggingface
S=/workspace/AsymGEMM-SFT-46/third_party/AsymGEMM/agent/anchors_tmp/stdtps96_status.log
DST=/scratch_local/user_data/shutian/kevin/cache/fused/gpt-oss-20b-bf16
if [ -f "$DST/model.safetensors.index.json" ] && [ -f "$DST/config.json" ]; then
  echo "G20-BUILD already present $(date '+%F %H:%M:%S')" >> "$S"; exit 0
fi
echo "G20-BUILD BEGIN $(date '+%F %H:%M:%S')" >> "$S"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN=""
CUDA_VISIBLE_DEVICES="" .venv/bin/python agent/anchors_tmp/gptoss20_dequant_c18.py
rc=$?
echo "G20-BUILD rc=$rc $(date '+%F %H:%M:%S')" >> "$S"
