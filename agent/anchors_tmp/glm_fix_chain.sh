#!/bin/bash
# Rerun the Flash asym dev cell (lost to the MLA-classifier FAIL) after the
# main GLM chain finishes. Fresh tag per stale-log discipline.
set -uo pipefail
export GPU=0 HOSTFLOOR=1300
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
for i in $(seq 1 1200); do grep -q "ALL-GLM-RUNS-COMPLETE" "$S" 2>/dev/null && break; sleep 30; done
grep -q "ALL-GLM-RUNS-COMPLETE" "$S" || { echo "GLMFIX-ABORT $(date +%H:%M)" >> "$S"; exit 1; }
MAX_STEPS=1 run_cell g3dev47a glm4.7-flash "asym_cpuadamwds|T1" 8000 "1"
echo "GLMFIX-DONE $(date +%H:%M)" >> "$S"
