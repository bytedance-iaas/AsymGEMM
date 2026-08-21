#!/bin/bash
# stdtps46_a2c.sh — Agent-2 continuation (supersedes stdtps46_a2.sh phases D/E
# and stdtps46_a2b.sh), reprioritized 06:4x: MAIN-FIGURE cells first, then
# upgrades, then lean-only fills. Serial, in-container, idempotent.
#   1  Mixtral-2r at the banked protocol's watchdog floor 35 (c12 2r campaign:
#      "floor 35 = sEP precedent"; the fused-path default 50 killed un@224k at
#      avail 49 GiB mid-step-2): un@224k (REQUIRED main cell), fsdp2@128k probe
#      (+160k/192k if it fits).
#   2  Mixtral-1r 288k decision cell checks: T2@288k (T1 fits but edge-taxed:
#      644 @95% vs banked T2@320k 670 — best-over-fitting-tiers rule) and the
#      un@288k GOOM confirmation on the pristine GPU 1.
#   3  Mixtral-1r zero3@64k (b2->b1) so the zero3 row is fully measured.
#   4  35B upgrades: 2r 256k best-over-batch (sEP-T2 b4->b3->b2, un b2, uo b2),
#      2r sEP-T2@384k b2 up-probe, 1r T1@640k tier check (+T1@768k if it fits).
#   5  lean-only fills: 35B-1r uo@640k/768k, Mixtral uo@160k both ranks.
set -uo pipefail
. /workspace/AsymGEMM-SFT-46/third_party/AsymGEMM/agent/anchors_tmp/stdtps46_lib.sh
POL="none|false|false|false|false|false"
Q35=q3.5-35b-a3b
MX=mixtral-8x22b
UN="superoffload_mem|unsloth-ohbm0"
UO="superoffload_mem|unsloth-off-ohbm0"
Z3="zero3_offload_mem|recomp"
FD="fsdp2_offload|recomp"
F35="HOST_MEM_WATCHDOG_FLOOR_GB=35"
MXARM="ASYM_ARENA_SHM_CAP_GB=285"
echo "=== STDTPS46-A2C BEGIN $(date '+%F %H:%M:%S') ===" >> "$S"

# ---- 1. Mixtral-2r @ floor 35 ----
echo "A2C-1 begin $(date +%H:%M)" >> "$S"
v=$(ARM_ENV="$F35" run_cell s2mx2un224f35 $MX "$UN" 224000 "1" "$POL" 2); echo "A2C-1 f35 un@224k 2r -> $v" >> "$S"
vf=$(ARM_ENV="$F35" run_cell s2mx2fd128f35 $MX "$FD" 128000 "1" "$POL" 2); echo "A2C-1 f35 fsdp2@128k 2r -> $vf" >> "$S"
if [ "$vf" = "TRAINED" ]; then
  vf=$(ARM_ENV="$F35" run_cell s2mx2fd160f35 $MX "$FD" 160000 "1" "$POL" 2); echo "A2C-1 f35 fsdp2@160k 2r -> $vf" >> "$S"
  [ "$vf" = "TRAINED" ] && { vf=$(ARM_ENV="$F35" run_cell s2mx2fd192f35 $MX "$FD" 192000 "1" "$POL" 2); echo "A2C-1 f35 fsdp2@192k 2r -> $vf" >> "$S"; }
  [ "$vf" = "TRAINED" ] && { vf=$(ARM_ENV="$F35" run_cell s2mx2fd224f35 $MX "$FD" 224000 "1" "$POL" 2); echo "A2C-1 f35 fsdp2@224k 2r -> $vf" >> "$S"; }
  [ "$vf" = "TRAINED" ] && { vf=$(ARM_ENV="$F35" run_cell s2mx2fd256f35 $MX "$FD" 256000 "1" "$POL" 2); echo "A2C-1 f35 fsdp2@256k 2r -> $vf" >> "$S"; }
  [ "$vf" = "TRAINED" ] && { vf=$(ARM_ENV="$F35" run_cell s2mx2fd288f35 $MX "$FD" 288000 "1" "$POL" 2); echo "A2C-1 f35 fsdp2@288k 2r -> $vf" >> "$S"; }
fi
echo "A2C-1-DONE $(date +%H:%M)" >> "$S"

# ---- 2. Mixtral-1r 288k decision-cell checks ----
echo "A2C-2 begin $(date +%H:%M)" >> "$S"
v=$(run_cell s2mxt2288 $MX "asym_cpuadamwds|T2" 288000 "1" "$POL" 1); echo "A2C-2 T2@288k 1r -> $v" >> "$S"
v=$(ONE_RANK_GPU=1 run_cell s2mxun288g1 $MX "$UN" 288000 "1" "$POL" 1); echo "A2C-2 un@288k 1r on GPU1 (confirm) -> $v" >> "$S"
echo "A2C-2-DONE $(date +%H:%M)" >> "$S"

# ---- 3. Mixtral-1r zero3@64k ----
v=$(run_cell s2mxz3064 $MX "$Z3" 64000 "2 1" "$POL" 1); echo "A2C-3 z3@64k 1r -> $v" >> "$S"
echo "A2C-3-DONE $(date +%H:%M)" >> "$S"

# ---- 4. 35B upgrades ----
echo "A2C-4 begin $(date +%H:%M)" >> "$S"
v=$(run_cell s2q35sep256 $Q35 "asym_sepplan2_cpuadamwds|T2" 256000 "4 3 2" "$POL" 2); echo "A2C-4 sep-T2@256k 2r walk -> $v" >> "$S"
v=$(run_cell s2q35un256 $Q35 "$UN" 256000 "2" "$POL" 2);  echo "A2C-4 un@256k 2r b2 -> $v" >> "$S"
v=$(run_cell s2q35uo256 $Q35 "$UO" 256000 "2" "$POL" 2);  echo "A2C-4 uo@256k 2r b2 -> $v" >> "$S"
v=$(run_cell s2q35sep384 $Q35 "asym_sepplan2_cpuadamwds|T2" 384000 "2" "$POL" 2); echo "A2C-4 sep-T2@384k 2r b2 -> $v" >> "$S"
v=$(run_cell s2q35t1640 $Q35 "asym_cpuadamwds|T1" 640000 "1" "$POL" 1); echo "A2C-4 T1@640k 1r -> $v" >> "$S"
if [ "$v" = "TRAINED" ]; then
  v=$(run_cell s2q35t1768 $Q35 "asym_cpuadamwds|T1" 768000 "1" "$POL" 1); echo "A2C-4 T1@768k 1r -> $v" >> "$S"
fi
echo "A2C-4-DONE $(date +%H:%M)" >> "$S"

# ---- 5. lean-only fills ----
echo "A2C-5 begin $(date +%H:%M)" >> "$S"
v=$(run_cell s2q35uo640 $Q35 "$UO" 640000 "1" "$POL" 1);  echo "A2C-5 35B uo@640k 1r -> $v" >> "$S"
v=$(run_cell s2q35uo768 $Q35 "$UO" 768000 "1" "$POL" 1);  echo "A2C-5 35B uo@768k 1r -> $v" >> "$S"
v=$(run_cell s2mxuo160 $MX "$UO" 160000 "1" "$POL" 1);    echo "A2C-5 MX uo@160k 1r -> $v" >> "$S"
v=$(ARM_ENV="$F35" run_cell s2mx2uo160f35 $MX "$UO" 160000 "1" "$POL" 2); echo "A2C-5 MX uo@160k 2r (f35) -> $v" >> "$S"
echo "A2C-5-DONE $(date +%H:%M)" >> "$S"
echo "=== STDTPS46-A2C ALL DONE $(date '+%F %H:%M:%S') ===" >> "$S"
