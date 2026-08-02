#!/bin/bash
# Air phase, SERIAL (replaces the killed parallel lanes): A1 (1-rank) all six
# rungs one-at-a-time on GPU0, then A2 (2-rank) six rungs on GPUs 0+1.
# Gated on F1-REDO-DONE.
set -uo pipefail
export GPU=0 HOSTFLOOR=1300 TORCHINDUCTOR_COMPILE_THREADS=1
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
for i in $(seq 1 2880); do grep -q "F2-PATCH2-DONE" "$S" 2>/dev/null && break; sleep 30; done
grep -q "F2-PATCH2-DONE" "$S" || { echo "A-SERIAL-ABORT-v2 $(date +%H:%M)" >> "$S"; exit 1; }
R=/workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/glmtp_rung.sh
echo "A1 begin $(date +%H:%M)" >> "$S"
GPU=0 HOSTFLOOR=1300 bash "$R" a1 glm4.5-air 128000 "3 2 1" 1
GPU=0 HOSTFLOOR=1300 bash "$R" a1 glm4.5-air 96000  "4 3 2" 1
GPU=0 HOSTFLOOR=1300 bash "$R" a1 glm4.5-air 64000  "6 4 3 2" 1
GPU=0 HOSTFLOOR=1300 bash "$R" a1 glm4.5-air 48000  "8 6 4" 1
GPU=0 HOSTFLOOR=1300 bash "$R" a1 glm4.5-air 32000  "12 8 6 4" 1
GPU=0 HOSTFLOOR=1300 bash "$R" a1 glm4.5-air 16000  "16 12 8" 1
echo "A1-DONE $(date +%H:%M)" >> "$S"
echo "A2 begin $(date +%H:%M)" >> "$S"
GPU="0,1" HOSTFLOOR=1300 GPU_POOL="0,1" bash "$R" a2 glm4.5-air 128000 "3 2 1" 2
GPU="0,1" HOSTFLOOR=1300 GPU_POOL="0,1" bash "$R" a2 glm4.5-air 96000  "4 3 2" 2
GPU="0,1" HOSTFLOOR=1300 GPU_POOL="0,1" bash "$R" a2 glm4.5-air 64000  "6 4 3 2" 2
GPU="0,1" HOSTFLOOR=1300 GPU_POOL="0,1" bash "$R" a2 glm4.5-air 48000  "8 6 4" 2
GPU="0,1" HOSTFLOOR=1300 GPU_POOL="0,1" bash "$R" a2 glm4.5-air 32000  "12 8 6 4" 2
GPU="0,1" HOSTFLOOR=1300 GPU_POOL="0,1" bash "$R" a2 glm4.5-air 16000  "16 12 8" 2
echo "A2-DONE GLMTP-ALL-DONE $(date +%H:%M)" >> "$S"
