#!/bin/bash
# stdtps46_a2b.sh — Agent-2 follow-up (runs AFTER stdtps46_a2.sh): completes
# the Mixtral-1r zero3 row so it becomes a MEASURED series (the standing
# "derived exception"): 64k b2->b1 (rc's banked b2 edge cell), and a 192k
# probe if 160k fit (rc-class wall is (128k,192k]; z3's own wall must be
# probed, not inherited). Idempotent like the main chain.
set -uo pipefail
. /workspace/AsymGEMM-SFT-46/third_party/AsymGEMM/agent/anchors_tmp/stdtps46_lib.sh
POL="none|false|false|false|false|false"
MX=mixtral-8x22b
Z3="zero3_offload_mem|recomp"
echo "=== STDTPS46-A2B BEGIN $(date '+%F %H:%M:%S') ===" >> "$S"
v=$(run_cell s2mxz3064 $MX "$Z3" 64000 "2 1" "$POL" 1); echo "A2B z3@64k -> $v" >> "$S"
# 192k probe only if the 160k cell trained (else 192k+ is beyond the measured wall)
if grep -q "^CELL s2mxz3160 .* -> TRAINED" "$S"; then
  v=$(run_cell s2mxz3192 $MX "$Z3" 192000 "1" "$POL" 1); echo "A2B z3@192k -> $v" >> "$S"
fi
echo "=== STDTPS46-A2B DONE $(date '+%F %H:%M:%S') ===" >> "$S"

# --- confirmation probes on the pristine GPU 1 (decision cells that GOOMed at the edge on GPU 0) ---
echo "=== STDTPS46-A2B CONFIRM BEGIN $(date '+%F %H:%M:%S') ===" >> "$S"
UN="superoffload_mem|unsloth-ohbm0"
v=$(ONE_RANK_GPU=1 run_cell s2mxun288g1 $MX "$UN" 288000 "1" "$POL" 1); echo "A2B confirm un@288k on GPU1 -> $v" >> "$S"
echo "=== STDTPS46-A2B CONFIRM DONE $(date '+%F %H:%M:%S') ===" >> "$S"

# --- tier cross-check at 288k-1r: T1 fits but edge-taxed (644 @95% HBM vs 802@256k;
#     banked T2@320k = 670). House rule = best over fitting tiers -> probe T2@288k.
echo "=== STDTPS46-A2B T2@288k BEGIN $(date '+%F %H:%M:%S') ===" >> "$S"
v=$(run_cell s2mxt2288 $MX "asym_cpuadamwds|T2" 288000 "1" "$POL" 1); echo "A2B T2@288k -> $v" >> "$S"
echo "=== STDTPS46-A2B T2@288k DONE $(date '+%F %H:%M:%S') ===" >> "$S"

# --- Mixtral-2r floor-35 reruns: the banked c12 2r cells (192k/256k fits, all
#     systems) ran at HOST_MEM_WATCHDOG_FLOOR_GB=35; the driver default for the
#     fused path is 50 and killed un@224k at avail 49 GiB mid-step-2. Re-measure
#     every 2r cell that did NOT train at floor 50 under the banked protocol.
echo "=== STDTPS46-A2B MX-2R FLOOR35 BEGIN $(date '+%F %H:%M:%S') ===" >> "$S"
F35="HOST_MEM_WATCHDOG_FLOOR_GB=35"
MXARM="ASYM_ARENA_SHM_CAP_GB=285"
chain_trained() { grep -q "^CELL $1 .* -> TRAINED" "$S"; }
chain_trained s2mx2un224 || { v=$(ARM_ENV="$F35" run_cell s2mx2un224f35 $MX "$UN" 224000 "1" "$POL" 2); echo "A2B f35 un@224k 2r -> $v" >> "$S"; }
chain_trained s2mx2t1224 || { v=$(ARM_ENV="$F35 $MXARM" run_cell s2mx2t1224f35 $MX "asym_sdp2_cpuadamwds|T1" 224000 "1" "$POL" 2); echo "A2B f35 sdp2-T1@224k 2r -> $v" >> "$S"; }
FD="fsdp2_offload|recomp"
if ! chain_trained s2mx2fd128; then
  vf=$(ARM_ENV="$F35" run_cell s2mx2fd128f35 $MX "$FD" 128000 "1" "$POL" 2); echo "A2B f35 fsdp2@128k 2r -> $vf" >> "$S"
  if [ "$vf" = "TRAINED" ]; then
    vf=$(ARM_ENV="$F35" run_cell s2mx2fd160f35 $MX "$FD" 160000 "1" "$POL" 2); echo "A2B f35 fsdp2@160k 2r -> $vf" >> "$S"
    [ "$vf" = "TRAINED" ] && { vf=$(ARM_ENV="$F35" run_cell s2mx2fd192f35 $MX "$FD" 192000 "1" "$POL" 2); echo "A2B f35 fsdp2@192k 2r -> $vf" >> "$S"; }
  fi
fi
echo "=== STDTPS46-A2B MX-2R FLOOR35 DONE $(date '+%F %H:%M:%S') ===" >> "$S"
