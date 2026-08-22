#!/bin/bash
# stdtps96_a4_g20.sh — 96G campaign, Agent 4, gpt-oss-20B: Phase A probes +
# Phase C remeasures. Reuse analysis (scratchpad reuse_gpt-oss-20b.txt):
# asym rows are reuse-valid (resv<=92G) at every grid rung except the two
# borderline T1@640k cells (98.1G 2r / 94.4G 1r); the 2r cap is 896k (T2B
# 86.8G reuse-valid) unless T1@1.02M unexpectedly squeezes under the
# occupier (185G resv 157G; T2B/T3@1.02M are HOST-C-OOM — host transfers).
# Baselines' 185G best-batch cells all ran >92G -> remeasured at b1 under
# the occupier on the grid 256/384/512/640/768/896k (walls in-grid probed;
# beyond-wall rungs red by monotonicity).
set -uo pipefail
. /workspace/AsymGEMM-SFT-46/third_party/AsymGEMM/agent/anchors_tmp/stdtps96_lib.sh
POL="none|false|false|false|false|false"
M=gpt-oss-20b
RC="superoffload_mem|recomp"
UN="superoffload_mem|unsloth-ohbm0"
UO="superoffload_mem|unsloth-off-ohbm0"
Z3="zero3_offload_mem|recomp"
FD="fsdp2_offload|recomp"
SEPL="asym_sepplanlink2_cpuadamwds"
start_occupiers || exit 1
echo "=== STDTPS96-A4-G20 BEGIN $(date '+%F %H:%M:%S') ===" >> "$S"

# ---- Phase A probes ----
v=$(run_cell s96g2t11024 $M "$SEPL|T1" 1024000 "1" "$POL" 2); echo "A4-G20 2r T1@1.02M (cap disproof; expect GOOM) -> $v" >> "$S"
if [ "$v" = "TRAINED" ]; then echo "A4-G20 UNEXPECTED: T1@1.02M fits at 96G — cap moves up; STOP and re-derive" >> "$S"; exit 0; fi
v=$(run_cell s96g2t1640 $M "$SEPL|T1" 640000 "1" "$POL" 2);  echo "A4-G20 2r T1@640k tier probe (185G resv 98.1) -> $v" >> "$S"
v=$(ONE_RANK_GPU=0 run_cell s96g1t1640 $M "asym_cpuadamwds|T1" 640000 "1" "$POL" 1); echo "A4-G20 1r T1@640k tier probe (185G resv 94.4) -> $v" >> "$S"
echo "A4-G20 PHASE-A DONE: cap=896k (grid 256..896k @128k) $(date +%H:%M)" >> "$S"

# ---- Phase C: 2-rank remeasures FIRST (grid columns; asym = reuse) ----
v=$(run_cell s96g2un256 $M "$UN" 256000 "1" "$POL" 2);  echo "A4-G20 2r un@256k -> $v" >> "$S"
v=$(run_cell s96g2un384 $M "$UN" 384000 "1" "$POL" 2);  echo "A4-G20 2r un@384k PROBE -> $v" >> "$S"
if [ "$v" = "TRAINED" ]; then
  v=$(run_cell s96g2un512 $M "$UN" 512000 "1" "$POL" 2); echo "A4-G20 2r un@512k PROBE -> $v" >> "$S"
fi
v=$(run_cell s96g2rc256 $M "$RC" 256000 "1" "$POL" 2);  echo "A4-G20 2r rc@256k PROBE (185G b1 115G) -> $v" >> "$S"
v=$(run_cell s96g2z3256 $M "$Z3" 256000 "1" "$POL" 2);  echo "A4-G20 2r z3@256k PROBE -> $v" >> "$S"
v=$(run_cell s96g2fd256 $M "$FD" 256000 "1" "$POL" 2);  echo "A4-G20 2r fd@256k PROBE -> $v" >> "$S"
v=$(run_cell s96g2uo256 $M "$UO" 256000 "1" "$POL" 2);  echo "A4-G20 2r uo@256k -> $v" >> "$S"
v=$(run_cell s96g2uo384 $M "$UO" 384000 "1" "$POL" 2);  echo "A4-G20 2r uo@384k PROBE -> $v" >> "$S"

# ---- Phase C: 1-rank remeasures on the same rungs ----
v=$(ONE_RANK_GPU=0 run_cell s96g1un256 $M "$UN" 256000 "1" "$POL" 1); echo "A4-G20 1r un@256k -> $v" >> "$S"
v=$(ONE_RANK_GPU=0 run_cell s96g1un384 $M "$UN" 384000 "1" "$POL" 1); echo "A4-G20 1r un@384k PROBE -> $v" >> "$S"
if [ "$v" = "TRAINED" ]; then
  v=$(ONE_RANK_GPU=0 run_cell s96g1un512 $M "$UN" 512000 "1" "$POL" 1); echo "A4-G20 1r un@512k PROBE -> $v" >> "$S"
fi
v=$(ONE_RANK_GPU=0 run_cell s96g1rc256 $M "$RC" 256000 "1" "$POL" 1); echo "A4-G20 1r rc@256k PROBE -> $v" >> "$S"
v=$(ONE_RANK_GPU=0 run_cell s96g1z3256 $M "$Z3" 256000 "1" "$POL" 1); echo "A4-G20 1r z3@256k PROBE -> $v" >> "$S"
v=$(ONE_RANK_GPU=0 run_cell s96g1fd256 $M "$FD" 256000 "1" "$POL" 1); echo "A4-G20 1r fd@256k PROBE -> $v" >> "$S"
v=$(ONE_RANK_GPU=0 run_cell s96g1uo256 $M "$UO" 256000 "1" "$POL" 1); echo "A4-G20 1r uo@256k -> $v" >> "$S"
v=$(ONE_RANK_GPU=0 run_cell s96g1uo384 $M "$UO" 384000 "1" "$POL" 1); echo "A4-G20 1r uo@384k PROBE -> $v" >> "$S"
echo "=== STDTPS96-A4-G20 ALL DONE $(date '+%F %H:%M:%S') ===" >> "$S"
