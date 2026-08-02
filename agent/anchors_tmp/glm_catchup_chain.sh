#!/bin/bash
# Final catch-up: Flash T3 192k same-workload cells (b2, b3) lost to the
# router-carve-out call-site race. Gated on GLM-WALKER-DONE.
set -uo pipefail
export GPU=0 HOSTFLOOR=1300
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
T3TOK="asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0"
for i in $(seq 1 1440); do grep -q "GLM-WALKER-DONE" "$S" 2>/dev/null && break; sleep 30; done
grep -q "GLM-WALKER-DONE" "$S" || { echo "CATCHUP-ABORT $(date +%H:%M)" >> "$S"; exit 1; }
run_cell g4val47t2 glm4.7-flash "$T3TOK" 192000 "2"
run_cell g4val47t3 glm4.7-flash "$T3TOK" 192000 "3"
echo "GLM-CATCHUP-DONE GLM-CAMPAIGN-ALL-CELLS-COMPLETE $(date +%H:%M)" >> "$S"
