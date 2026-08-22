#!/bin/bash
# stdtps96_a2_phaseC_2r_b.sh — Phase C 2r resume after the 384K incident:
# GPU3 occupier re-tightened (budget was +2.1G open since 11:37 — audit:
# NO verdict flips, all TRAINED cells <=93.6G, all OOMs OOMed despite the
# slack); the T1@384 SIGABRT cascade discarded per the fail-stop rule.
# This chain: T1@384 RETRY (one repeat allowed for the NCCL-flake class;
# a second SIGABRT fail-stops the rung for runtime investigation), then
# the ladder ONLY on GOOM/COOM, then the Flash baselines + uo walk.
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
echo "== stdtps96 A2 phaseC 2r-b resume $(date +%F_%H:%M) occ='$OCC_PIDS' ==" >> "$S"

# 384K rung, fail-stop discipline
v=$(ARM_ENV="$AARM" run_cell s96fl2f384a2 glm4.7-flash "$SDP|T1" 384000 "1" "$POL" 2)
case "$v" in
  TRAINED) : ;;
  GOOM|COOM)
    v=$(ARM_ENV="$AARM" run_cell s96fl2f384b2 glm4.7-flash "$SDP|T2" 384000 "1" "$POL" 2)
    case "$v" in
      TRAINED) : ;;
      GOOM|COOM) ARM_ENV="$AARM" run_cell s96fl2f384c2 glm4.7-flash "$SDP|T3" 384000 "1" "$POL" 2 ;;
      *) echo "FAILSTOP s96fl2f384b2 -> $v (no demotion)" >> "$S" ;;
    esac ;;
  *) echo "FAILSTOP s96fl2f384a2 -> $v (second non-OOM failure; rung parked for runtime investigation)" >> "$S" ;;
esac

# Flash baselines at the low rungs
ARM_ENV="" run_cell s96fl2r192 glm4.7-flash "$RC" 192000 "1" "$POL" 2
ARM_ENV="" run_cell s96fl2d192 glm4.7-flash "$F2" 192000 "1" "$POL" 2
ARM_ENV="" run_cell s96fl2z192 glm4.7-flash "$Z3" 192000 "1" "$POL" 2
v=$(ARM_ENV="" run_cell s96fl2u192 glm4.7-flash "$UNS" 192000 "2 1" "$POL" 2)
[ "$v" = "TRAINED" ] && ARM_ENV="" run_cell s96fl2u256 glm4.7-flash "$UNS" 256000 "1" "$POL" 2
for sq in 192 256 320 384 448; do
  v=$(ARM_ENV="" run_cell "s96fl2o${sq}" glm4.7-flash "$UO" "${sq}000" "1" "$POL" 2)
  [ "$v" != "TRAINED" ] && break
done
echo "== stdtps96 A2 phaseC 2r-b DONE $(date +%F_%H:%M) ==" >> "$S"
