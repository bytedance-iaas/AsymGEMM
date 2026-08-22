#!/bin/bash
cd /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM
timeout 25 .venv/bin/python agent/anchors_tmp/hbm96_occupy.py --device 0 &
OP=$!
sleep 15
echo "--- nvidia-smi during occupy ---"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | head -1
wait $OP 2>/dev/null
echo "--- after exit ---"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -1
