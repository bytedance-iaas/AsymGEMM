#!/bin/bash
# fig12 probe 2 (v2) — GLM-4.7-Flash shipped-T3 parity pair (A vs staged/reaim B)
# at 96k, plus 30B 320k parity pair (the old figure's length).
# Usage: GPU=0 HOSTFLOOR=600 bash fig12_probe2.sh
set -uo pipefail
export GPU="${GPU:-0}" HOSTFLOOR="${HOSTFLOOR:-600}" GPU_POOL="${GPU:-0}"
. /workspace/AsymGEMM-SFT/third_party/AsymGEMM/agent/anchors_tmp/fig12_lib.sh
POL="none|false|true|false|false|false"
SYS="asym_cpuadamwds|T3"

echo "PROBE2 begin $(date +%H:%M)" >> "$S"
# GLM Flash 96k: A then B(staged) then B(reaim), shipped config
ARM_ENV=""                            run_cell g2a96 glm4.7-flash "$SYS" 96000 "4 3 2" "$POL" 1
ARM_ENV="ASYMM_LORA_KERNELS=staged"   run_cell g2s96 glm4.7-flash "$SYS" 96000 "4 3 2" "$POL" 1
ARM_ENV="ASYMM_LORA_KERNELS=reaim"    run_cell g2r96 glm4.7-flash "$SYS" 96000 "4 3 2" "$POL" 1
# 30B 320k b1 (old figure length): A then B(staged)
ARM_ENV=""                            run_cell k2a320 q3-30b-a3b "$SYS" 320000 "1" "$POL" 1
ARM_ENV="ASYMM_LORA_KERNELS=staged"   run_cell k2s320 q3-30b-a3b "$SYS" 320000 "1" "$POL" 1
echo "PROBE2-DONE $(date +%H:%M)" >> "$S"
