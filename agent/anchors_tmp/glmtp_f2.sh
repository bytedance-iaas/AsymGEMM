#!/bin/bash
# Flash 2-rank (streaming-EP/DP over GPUs 0+1), serial. Gated on both F1 lanes.
set -uo pipefail
export GPU="0,1" HOSTFLOOR=1200 TORCHINDUCTOR_COMPILE_THREADS=1 GPU_POOL="0,1"
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
for i in $(seq 1 2880); do grep -q "F1-LANEA-DONE" "$S" 2>/dev/null && grep -q "F1-LANEB-DONE" "$S" 2>/dev/null && break; sleep 30; done
{ grep -q "F1-LANEA-DONE" "$S" && grep -q "F1-LANEB-DONE" "$S"; } || { echo "F2-ABORT $(date +%H:%M)" >> "$S"; exit 1; }
echo "F2 begin $(date +%H:%M)" >> "$S"
R=/workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/glmtp_rung.sh
GPU="0,1" HOSTFLOOR=1200 GPU_POOL="0,1" bash "$R" f2 glm4.7-flash 192000 "4 2 1" 2
GPU="0,1" HOSTFLOOR=1200 GPU_POOL="0,1" bash "$R" f2 glm4.7-flash 160000 "2 1" 2
GPU="0,1" HOSTFLOOR=1200 GPU_POOL="0,1" bash "$R" f2 glm4.7-flash 128000 "3 2 1" 2
GPU="0,1" HOSTFLOOR=1200 GPU_POOL="0,1" bash "$R" f2 glm4.7-flash 96000  "4 3 2" 2
GPU="0,1" HOSTFLOOR=1200 GPU_POOL="0,1" bash "$R" f2 glm4.7-flash 64000  "6 4 3 2" 2
GPU="0,1" HOSTFLOOR=1200 GPU_POOL="0,1" bash "$R" f2 glm4.7-flash 32000  "12 8 6 4" 2
echo "F2-DONE $(date +%H:%M)" >> "$S"
