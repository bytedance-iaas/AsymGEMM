#!/bin/bash
# Air 1-rank lane B (GPU1): 96k, 64k, 32k. Gated on F2-DONE.
set -uo pipefail
export GPU=1 HOSTFLOOR=600 TORCHINDUCTOR_COMPILE_THREADS=1
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
for i in $(seq 1 2880); do grep -q "F2-DONE" "$S" 2>/dev/null && break; sleep 30; done
grep -q "F2-DONE" "$S" || { echo "A1-LANEB-ABORT $(date +%H:%M)" >> "$S"; exit 1; }
echo "A1-LANEB begin $(date +%H:%M)" >> "$S"
R=/workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/glmtp_rung.sh
GPU=1 HOSTFLOOR=600 bash "$R" a1 glm4.5-air 96000 "4 3 2" 1
GPU=1 HOSTFLOOR=600 bash "$R" a1 glm4.5-air 64000 "6 4 3 2" 1
GPU=1 HOSTFLOOR=600 bash "$R" a1 glm4.5-air 32000 "12 8 6 4" 1
echo "A1-LANEB-DONE $(date +%H:%M)" >> "$S"
