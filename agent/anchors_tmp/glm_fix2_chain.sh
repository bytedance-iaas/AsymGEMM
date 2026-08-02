#!/bin/bash
# Rerun the Air asym dev cell (lost to the DS-V3 router-swap bug) after
# glm_fix_chain's Flash dev rerun. Fresh tag per stale-log discipline.
set -uo pipefail
export GPU=0 HOSTFLOOR=1300
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
for i in $(seq 1 1200); do grep -q "GLMFIX-DONE" "$S" 2>/dev/null && break; sleep 30; done
grep -q "GLMFIX-DONE" "$S" || { echo "GLMFIX2-ABORT $(date +%H:%M)" >> "$S"; exit 1; }
MAX_STEPS=1 run_cell g3dev45a glm4.5-air "asym_cpuadamwds|T1" 8000 "1"
echo "GLMFIX2-DONE ALL-GLM-DEV-RERUNS-COMPLETE $(date +%H:%M)" >> "$S"
