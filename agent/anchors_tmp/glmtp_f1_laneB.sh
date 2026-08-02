#!/bin/bash
# Flash 1-rank, lane B (GPU1): rungs 160k, 128k, 64k.
set -uo pipefail
export GPU=1 HOSTFLOOR=500 TORCHINDUCTOR_COMPILE_THREADS=1
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
echo "F1-LANEB begin $(date +%H:%M)" >> "$S"
R=/workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/glmtp_rung.sh
GPU=1 HOSTFLOOR=500 bash "$R" f1 glm4.7-flash 160000 "2 1" 1
GPU=1 HOSTFLOOR=500 bash "$R" f1 glm4.7-flash 128000 "3 2 1" 1
GPU=1 HOSTFLOOR=500 bash "$R" f1 glm4.7-flash 64000  "6 4 3 2" 1
echo "F1-LANEB-DONE $(date +%H:%M)" >> "$S"
