#!/bin/bash
# Air 2-rank, serial over GPUs 0+1. Gated on both A1 lanes.
set -uo pipefail
export GPU="0,1" HOSTFLOOR=1200 TORCHINDUCTOR_COMPILE_THREADS=1 GPU_POOL="0,1"
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
for i in $(seq 1 2880); do grep -q "A1-LANEA-DONE" "$S" 2>/dev/null && grep -q "A1-LANEB-DONE" "$S" 2>/dev/null && break; sleep 30; done
{ grep -q "A1-LANEA-DONE" "$S" && grep -q "A1-LANEB-DONE" "$S"; } || { echo "A2-ABORT $(date +%H:%M)" >> "$S"; exit 1; }
echo "A2 begin $(date +%H:%M)" >> "$S"
R=/workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/glmtp_rung.sh
GPU="0,1" HOSTFLOOR=1200 GPU_POOL="0,1" bash "$R" a2 glm4.5-air 128000 "3 2 1" 2
GPU="0,1" HOSTFLOOR=1200 GPU_POOL="0,1" bash "$R" a2 glm4.5-air 96000  "4 3 2" 2
GPU="0,1" HOSTFLOOR=1200 GPU_POOL="0,1" bash "$R" a2 glm4.5-air 64000  "6 4 3 2" 2
GPU="0,1" HOSTFLOOR=1200 GPU_POOL="0,1" bash "$R" a2 glm4.5-air 48000  "8 6 4" 2
GPU="0,1" HOSTFLOOR=1200 GPU_POOL="0,1" bash "$R" a2 glm4.5-air 32000  "12 8 6 4" 2
GPU="0,1" HOSTFLOOR=1200 GPU_POOL="0,1" bash "$R" a2 glm4.5-air 16000  "16 12 8" 2
echo "A2-DONE GLMTP-ALL-DONE $(date +%H:%M)" >> "$S"
