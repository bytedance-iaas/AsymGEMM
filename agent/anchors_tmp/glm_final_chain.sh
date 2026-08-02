#!/bin/bash
# Final GLM cell: Flash asym dev, attempt 3 (after the selection-parser MLA
# fix). Gated on GLM-CATCHUP-DONE (end of the entire queue).
set -uo pipefail
export GPU=0 HOSTFLOOR=1300
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
for i in $(seq 1 1440); do grep -q "GLM-CATCHUP-DONE" "$S" 2>/dev/null && break; sleep 30; done
grep -q "GLM-CATCHUP-DONE" "$S" || { echo "FINAL-ABORT $(date +%H:%M)" >> "$S"; exit 1; }
MAX_STEPS=1 run_cell g4dev47a glm4.7-flash "asym_cpuadamwds|T1" 8000 "1"
echo "GLM-FINAL-DONE EVERYTHING-COMPLETE $(date +%H:%M)" >> "$S"
