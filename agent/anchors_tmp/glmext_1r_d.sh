#!/bin/bash
# glmext_1r_d.sh — Flash 1r solo-column, dial notch 2 (2026-08-07).
# T3-ohbm4 @1152k GOOM'd by ~1.3 GiB (166.6 in use + 17.6 alloc vs 184 cap;
# roots@N=4 ≈ 55 GB -> peak ≈ 184). ohbm5 ≈ 44 GB roots -> peak ≈ 173 GiB
# (~11 GiB slack) with 44 GB host relief vs the ~10 GB host shortfall
# (T3-ohbm0 died at avail 34 < floor 35; this runs floor 25 like the
# fairness-confirmed uo re-probe). If TRAINED -> the 1280k pair.
set -uo pipefail
export TORCHINDUCTOR_COMPILE_THREADS=1
export GPU=0 HOSTFLOOR=500
export HOST_MEM_WATCHDOG_FLOOR_GB=25
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
echo "GLM1RE begin (ohbm6) $(date +%H:%M)" >> "$S"
P="none|false|false|false|false|false"
export CUDA_VISIBLE_DEVICES=0
T3O="asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm6"

v=$(run_cell x1et3o1152 glm4.7-flash "$T3O" 1152000 "1" "$P" 1)
echo "SOLO-1152 verdict: $v $(date +%H:%M)" >> "$S"
if [ "$v" = "TRAINED" ]; then
  run_cell x1euo1280b glm4.7-flash "superoffload_mem|unsloth-off-ohbm0" 1280000 "1" "$P" 1 >/dev/null
  v2=$(run_cell x1et3o1280 glm4.7-flash "$T3O" 1280000 "1" "$P" 1)
  echo "SOLO-1280 verdict: $v2 $(date +%H:%M)" >> "$S"
fi
echo "X1E2-DONE $(date +%H:%M)" >> "$S"
