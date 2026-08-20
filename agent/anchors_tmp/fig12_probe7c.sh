#!/bin/bash
# fig12 probe 7c — Mixtral 2nd+3rd lengths: 32k b4 and 128k b1 pairs.
set -uo pipefail
export GPU="${GPU:-0}" HOSTFLOOR="${HOSTFLOOR:-500}" GPU_POOL="${GPU:-0}"
. /workspace/AsymGEMM-SFT/third_party/AsymGEMM/agent/anchors_tmp/fig12_lib.sh
POLN="none|false|false|false|false|false"
echo "PROBE7C begin $(date +%H:%M)" >> "$S"
ARM_ENV="" run_cell p7nmx32 mixtral-8x22b "asym_cpuadamwds|unsloth" 32000 "4 2" "$POLN" 1
ARM_ENV="" run_cell p7amx32 mixtral-8x22b "asym_cpuadamwds|T1" 32000 "4 2" "$POLN" 1
ARM_ENV="" run_cell p7nmx128 mixtral-8x22b "asym_cpuadamwds|unsloth" 128000 "1" "$POLN" 1
ARM_ENV="" run_cell p7amx128 mixtral-8x22b "asym_cpuadamwds|T1" 128000 "1" "$POLN" 1
echo "PROBE7C-DONE $(date +%H:%M)" >> "$S"
