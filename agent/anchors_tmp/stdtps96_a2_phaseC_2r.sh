#!/bin/bash
# stdtps96_a2_phaseC_2r.sh — Agent 2 Phase C, 2-RANK fills.
# Air grid 32/64/96/128/160/192K (cap cell 208K banked): asym T1 low rungs
# (T1 fits <=112K: s96air2c112_t1) + banked T2 @128/160/192 from Phase A
# (b2 up-probes added); baselines walked to their measured 96G walls.
# Flash grid 192/256/320/384/448/512K: asym 448(T2)/512(T3) banked; ladder
# fills 192-384; baseline walls probed at the lowest rungs.
# uo (unsloth_off) measured b1-only (banked, MAIN-dropped).
set -u
export GPU="0,1" HOSTFLOOR=500
source /workspace/AsymGEMM-SFT/third_party/AsymGEMM/agent/anchors_tmp/stdtps96_lib.sh
export GPU_POOL="0,1" DDP_TIMEOUT=1500
POL="none|false|false|false|false|false"
SDP="asym_sdp2_cpuadamwds"
UNS="superoffload_mem|unsloth"
RC="superoffload_mem|recomp"
UO="superoffload_mem|unsloth-off"
Z3="zero3_offload_mem|recomp"
F2="fsdp2_offload|recomp"
AARM="ASYM_ARENA_SHM_CAP_GB=400"
echo "== stdtps96 A2 phaseC 2r start $(date +%F_%H:%M) ==" >> "$S"

# ---------- GLM-4.5-Air ----------
# asym low rungs (T1; b-walks halved from 185G seeds)
ARM_ENV="$AARM" run_cell s96air2f32a  glm4.5-air "$SDP|T1" 32000 "4 2 1" "$POL" 2
ARM_ENV="$AARM" run_cell s96air2f64a  glm4.5-air "$SDP|T1" 64000 "2 1" "$POL" 2
ARM_ENV="$AARM" run_cell s96air2f96a  glm4.5-air "$SDP|T1" 96000 "2 1" "$POL" 2
# b2 up-probes on the banked T2 rungs (max-TP rule)
ARM_ENV="$AARM" run_cell s96air2f128p glm4.5-air "$SDP|T2" 128000 "2" "$POL" 2
ARM_ENV="$AARM" run_cell s96air2f160p glm4.5-air "$SDP|T2" 160000 "2" "$POL" 2
ARM_ENV="$AARM" run_cell s96air2f192p glm4.5-air "$SDP|T2" 192000 "2" "$POL" 2
# recomp: prior wall 48-64K -> walk up to first GOOM
v=$(ARM_ENV="" run_cell s96air2r32 glm4.5-air "$RC" 32000 "4 2 1" "$POL" 2)
if [ "$v" = "TRAINED" ]; then
  v=$(ARM_ENV="" run_cell s96air2r64 glm4.5-air "$RC" 64000 "2 1" "$POL" 2)
  [ "$v" = "TRAINED" ] && ARM_ENV="" run_cell s96air2r96 glm4.5-air "$RC" 96000 "1" "$POL" 2
fi
# unsloth: prior wall 80-96K
v=$(ARM_ENV="" run_cell s96air2u32 glm4.5-air "$UNS" 32000 "4 2 1" "$POL" 2)
if [ "$v" = "TRAINED" ]; then
  v=$(ARM_ENV="" run_cell s96air2u64 glm4.5-air "$UNS" 64000 "2 1" "$POL" 2)
  if [ "$v" = "TRAINED" ]; then
    v=$(ARM_ENV="" run_cell s96air2u96 glm4.5-air "$UNS" 96000 "1" "$POL" 2)
    [ "$v" = "TRAINED" ] && ARM_ENV="" run_cell s96air2u128 glm4.5-air "$UNS" 128000 "1" "$POL" 2
  fi
fi
# zero3 mirrors rc rungs; fsdp2 walks its own wall
v=$(ARM_ENV="" run_cell s96air2z32 glm4.5-air "$Z3" 32000 "2 1" "$POL" 2)
[ "$v" = "TRAINED" ] && ARM_ENV="" run_cell s96air2z64 glm4.5-air "$Z3" 64000 "1" "$POL" 2
v=$(ARM_ENV="" run_cell s96air2d32 glm4.5-air "$F2" 32000 "2 1" "$POL" 2)
[ "$v" = "TRAINED" ] && ARM_ENV="" run_cell s96air2d64 glm4.5-air "$F2" 64000 "1" "$POL" 2
# uns-off b1-only until first dead
for sq in 32 64 96 128 160; do
  v=$(ARM_ENV="" run_cell "s96air2o${sq}" glm4.5-air "$UO" "${sq}000" "1" "$POL" 2)
  [ "$v" != "TRAINED" ] && break
done

# ---------- GLM-4.7-Flash ----------
# asym fills 192-384: ladder T1 -> T2 -> T3 per rung (T1 96G wall unknown)
for sq in 192 256 320 384; do
  v=$(ARM_ENV="$AARM" run_cell "s96fl2f${sq}a" glm4.7-flash "$SDP|T1" "${sq}000" "1" "$POL" 2)
  if [ "$v" != "TRAINED" ]; then
    v=$(ARM_ENV="$AARM" run_cell "s96fl2f${sq}b" glm4.7-flash "$SDP|T2" "${sq}000" "1" "$POL" 2)
    [ "$v" != "TRAINED" ] && ARM_ENV="$AARM" run_cell "s96fl2f${sq}c" glm4.7-flash "$SDP|T3" "${sq}000" "1" "$POL" 2
  fi
done
# baselines at the low rungs: rc/fd/z3 expect dead @192 (measure the all-OOM);
# uns expect fit @192, dead by 256; uo may run deep (b1-only walk)
ARM_ENV="" run_cell s96fl2r192 glm4.7-flash "$RC" 192000 "1" "$POL" 2
ARM_ENV="" run_cell s96fl2d192 glm4.7-flash "$F2" 192000 "1" "$POL" 2
ARM_ENV="" run_cell s96fl2z192 glm4.7-flash "$Z3" 192000 "1" "$POL" 2
v=$(ARM_ENV="" run_cell s96fl2u192 glm4.7-flash "$UNS" 192000 "2 1" "$POL" 2)
[ "$v" = "TRAINED" ] && ARM_ENV="" run_cell s96fl2u256 glm4.7-flash "$UNS" 256000 "1" "$POL" 2
for sq in 192 256 320 384 448; do
  v=$(ARM_ENV="" run_cell "s96fl2o${sq}" glm4.7-flash "$UO" "${sq}000" "1" "$POL" 2)
  [ "$v" != "TRAINED" ] && break
done
echo "== stdtps96 A2 phaseC 2r DONE $(date +%F_%H:%M) ==" >> "$S"
