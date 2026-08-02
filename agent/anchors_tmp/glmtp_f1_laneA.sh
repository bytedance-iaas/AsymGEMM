#!/bin/bash
# Flash 1-rank, lane A (GPU0): rungs 192k, 96k, 32k.
set -uo pipefail
export GPU=0 HOSTFLOOR=500 TORCHINDUCTOR_COMPILE_THREADS=1
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
echo "F1-LANEA begin $(date +%H:%M)" >> "$S"
R=/workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/glmtp_rung.sh
GPU=0 HOSTFLOOR=500 bash "$R" f1 glm4.7-flash 192000 "4 2 1" 1
GPU=0 HOSTFLOOR=500 bash "$R" f1 glm4.7-flash 96000  "4 3 2" 1
GPU=0 HOSTFLOOR=500 bash "$R" f1 glm4.7-flash 32000  "12 8 6 4" 1
echo "F1-LANEA-DONE $(date +%H:%M)" >> "$S"
