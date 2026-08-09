#!/bin/bash
# glmext_1r_e.sh — Flash 1r solo-column round E (2026-08-07, see run_glms.md
# §Log claim). 1088k pair first (verdict-pure), ohbm8@1152k fallback.
set -uo pipefail
export TORCHINDUCTOR_COMPILE_THREADS=1
export GPU=0 HOSTFLOOR=500
export HOST_MEM_WATCHDOG_FLOOR_GB=25
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
echo "GLM1RE begin $(date +%H:%M)" >> "$S"
P="none|false|false|false|false|false"
export CUDA_VISIBLE_DEVICES=0
BE=asym_cpuadamwds
T3RAW="${BE}|recomp-off-full-fg-ker000-ceil0000-ohbm0"

vuo=$(run_cell x1euo1088 glm4.7-flash "superoffload_mem|unsloth-off-ohbm0" 1088000 "1" "$P" 1)
echo "E-UO-1088: $vuo $(date +%H:%M)" >> "$S"
if [ "$vuo" != "TRAINED" ]; then
  v=$(run_cell x1et3r1088 glm4.7-flash "$T3RAW" 1088000 "1" "$P" 1)
  echo "E-SOLO-1088: $v $(date +%H:%M)" >> "$S"
else
  v=$(run_cell x1et3o8_1152 glm4.7-flash "${BE}|recomp-off-full-fg-ker000-ceil0000-ohbm8" 1152000 "1" "$P" 1)
  echo "E-SOLO-1152-ohbm8: $v $(date +%H:%M)" >> "$S"
fi
echo "X1E2-DONE $(date +%H:%M)" >> "$S"
