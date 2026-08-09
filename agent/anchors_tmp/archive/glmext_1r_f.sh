#!/bin/bash
# glmext_1r_f.sh — Flash 1r solo-column round F (see run_glms.md claim).
set -uo pipefail
export TORCHINDUCTOR_COMPILE_THREADS=1
export GPU=0 HOSTFLOOR=500
export HOST_MEM_WATCHDOG_FLOOR_GB=25
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
echo "GLM1RF begin $(date +%H:%M)" >> "$S"
P="none|false|false|false|false|false"
export CUDA_VISIBLE_DEVICES=0
BE=asym_cpuadamwds

v=$(run_cell x1ft3o8_1088 glm4.7-flash "${BE}|recomp-off-full-fg-ker000-ceil0000-ohbm8" 1088000 "1" "$P" 1)
echo "F-SOLO-1088-ohbm8: $v $(date +%H:%M)" >> "$S"
if [ "$v" != "TRAINED" ]; then
  vuo=$(run_cell x1fuo1056 glm4.7-flash "superoffload_mem|unsloth-off-ohbm0" 1056000 "1" "$P" 1)
  echo "F-UO-1056: $vuo $(date +%H:%M)" >> "$S"
  if [ "$vuo" != "TRAINED" ]; then
    v2=$(run_cell x1ft3r1056 glm4.7-flash "${BE}|recomp-off-full-fg-ker000-ceil0000-ohbm0" 1056000 "1" "$P" 1)
    echo "F-SOLO-1056: $v2 $(date +%H:%M)" >> "$S"
  fi
fi
echo "X1F-DONE $(date +%H:%M)" >> "$S"
