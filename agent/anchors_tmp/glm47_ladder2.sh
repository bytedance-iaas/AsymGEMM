#!/bin/bash
# Ladder round 2a: verify the MLA share-dedupe on the plain T3 config
# (memory should hold or improve; host D2H halves on attention inputs).
set -uo pipefail
export GPU=0 HOSTFLOOR=1300
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
T3TOK="asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0"
for i in $(seq 1 1440); do grep -q "GLM47-LADDER1-DONE" "$S" 2>/dev/null && break; sleep 30; done
grep -q "GLM47-LADDER1-DONE" "$S" || { echo "LADDER2-ABORT $(date +%H:%M)" >> "$S"; exit 1; }
run_cell l47dedup glm4.7-flash "$T3TOK" 192000 "5"
echo "GLM47-LADDER2-DONE $(date +%H:%M)" >> "$S"
