#!/bin/bash
# Air 1-rank lane A (GPU0): 128k, 48k, 16k. Gated on F2-DONE.
set -uo pipefail
export GPU=0 HOSTFLOOR=600 TORCHINDUCTOR_COMPILE_THREADS=1
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
for i in $(seq 1 2880); do grep -q "F2-DONE" "$S" 2>/dev/null && break; sleep 30; done
grep -q "F2-DONE" "$S" || { echo "A1-LANEA-ABORT $(date +%H:%M)" >> "$S"; exit 1; }
echo "A1-LANEA begin $(date +%H:%M)" >> "$S"
R=/workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/glmtp_rung.sh
GPU=0 HOSTFLOOR=600 bash "$R" a1 glm4.5-air 128000 "3 2 1" 1
GPU=0 HOSTFLOOR=600 bash "$R" a1 glm4.5-air 48000  "8 6 4" 1
GPU=0 HOSTFLOOR=600 bash "$R" a1 glm4.5-air 16000  "16 12 8" 1
echo "A1-LANEA-DONE $(date +%H:%M)" >> "$S"
