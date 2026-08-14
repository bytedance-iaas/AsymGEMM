#!/bin/bash
# fig12 probe 4 — staged-arm NaN debug at the cheap cell (30B 96k b8, T3).
set -uo pipefail
export GPU="${GPU:-0}" HOSTFLOOR="${HOSTFLOOR:-600}" GPU_POOL="${GPU:-0}"
. /workspace/AsymGEMM-SFT/third_party/AsymGEMM/agent/anchors_tmp/fig12_lib.sh
POL="none|false|true|false|false|false"
SYS="asym_cpuadamwds|T3"
echo "PROBE4 begin $(date +%H:%M)" >> "$S"
ARM_ENV="ASYMM_LORA_KERNELS=staged ASYMM_LORA_KERNELS_DEBUG=1" run_cell kfs96 q3-30b-a3b "$SYS" 96000 "8" "$POL" 1
echo "PROBE4-DONE $(date +%H:%M)" >> "$S"
