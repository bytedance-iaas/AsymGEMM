#!/bin/bash
set -uo pipefail
export GPU="${GPU:-0}" HOSTFLOOR="${HOSTFLOOR:-600}" GPU_POOL="${GPU:-0}"
. /workspace/AsymGEMM-SFT/third_party/AsymGEMM/agent/anchors_tmp/fig12_lib.sh
POLA="none|false|true|false|false|false"
echo "MERGESMOKE begin $(date +%H:%M)" >> "$S"
ARM_ENV="" run_cell mrg819 q3-30b-a3b "asym_cpuadamwds|T3" 96000 "8" "$POLA" 1
echo "MERGESMOKE-DONE $(date +%H:%M)" >> "$S"
