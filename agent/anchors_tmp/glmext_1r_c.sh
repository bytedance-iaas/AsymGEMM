#!/bin/bash
# glmext_1r_c.sh — Flash 1r FINAL round (2026-08-07): the 1152k solo column.
# Autopsy: both 1152k COOMs were genuine host exhaustion (avail 34 < floor 35,
# RSS climbing mid-step) — but asym T3 died holding ~60 GiB FREE HBM (125/188).
# Fix = asym's own placement flexibility: T3-ohbm4 keeps every 4th outer
# checkpoint root in HBM (~55 GB host relief into free HBM; ohbm grammar is
# UNSLOTH_GC_OUTER_HBM_EVERY_N, llama-panel precedent +ohbm12). FAIRNESS: the
# floor must match on both sides -> BOTH contenders rerun at floor 25 (the
# FSDP-campaign c14 floor precedent): uo@1152k re-probe (expect COOM again —
# it was mid-climb with no HBM headroom to trade) THEN T3-ohbm4@1152k. If the
# T3-ohbm4 fits, 1280k gets the same pair treatment for a second solo column.
set -uo pipefail
export TORCHINDUCTOR_COMPILE_THREADS=1
export GPU=0 HOSTFLOOR=500
export HOST_MEM_WATCHDOG_FLOOR_GB=25
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
echo "GLM1RC begin (floor25 fairness pair) $(date +%H:%M)" >> "$S"
P="none|false|false|false|false|false"
export CUDA_VISIBLE_DEVICES=0
BE=asym_cpuadamwds
T3O="${BE}|recomp-off-full-fg-ker000-ceil0000-ohbm4"

# 1) uo re-probe at floor 25 (fairness twin of the asym attempt)
run_cell x1cuo1152 glm4.7-flash "superoffload_mem|unsloth-off-ohbm0" 1152000 "1" "$P" 1 >/dev/null

# 2) asym T3-ohbm4 at 1152k (floor 25)
v=$(run_cell x1ct3o1152 glm4.7-flash "$T3O" 1152000 "1" "$P" 1)
echo "SOLO-1152 verdict: $v $(date +%H:%M)" >> "$S"

# 3) if solo landed, the 1280k pair for the second solo column
if [ "$v" = "TRAINED" ]; then
  run_cell x1cuo1280 glm4.7-flash "superoffload_mem|unsloth-off-ohbm0" 1280000 "1" "$P" 1 >/dev/null
  v2=$(run_cell x1ct3o1280 glm4.7-flash "$T3O" 1280000 "1" "$P" 1)
  echo "SOLO-1280 verdict: $v2 $(date +%H:%M)" >> "$S"
fi
echo "X1C-DONE $(date +%H:%M)" >> "$S"
