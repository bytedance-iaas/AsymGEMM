#!/bin/bash
# glmext_1r_g.sh — Option A (2026-08-07): asym ceiling certificate @1152k.
# T3-ohbm8: HBM ≈ 184-186 (razor) · host ≈ 940 RSS — fits => crown 1.15M;
# dies => 1.09M ceiling FINAL (T2 G-OOM + T3-raw C-OOM + this = full ladder).
set -uo pipefail
export TORCHINDUCTOR_COMPILE_THREADS=1
export GPU=0 HOSTFLOOR=500
export HOST_MEM_WATCHDOG_FLOOR_GB=25
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
echo "GLM1RG begin (Option A) $(date +%H:%M)" >> "$S"
P="none|false|false|false|false|false"
export CUDA_VISIBLE_DEVICES=0
v=$(run_cell x1gt3o8_1152 glm4.7-flash "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm8" 1152000 "1" "$P" 1)
echo "OPTION-A-1152: $v $(date +%H:%M)" >> "$S"
echo "X1G-DONE $(date +%H:%M)" >> "$S"
