#!/bin/bash
# stdtps96_a2_phaseA.sh — Agent 2 (GLMs) Phase A: 2r ceiling search under
# the 96G occupiers (standardize_tps_96gb.md). Runs INSIDE the container on
# the simulated pair (inside ids 0,1). Walks the asym ladder at b1 from the
# ceiling prior; up-steps while TRAINED; on fail tries the next legal tier
# (GLM ladder T1 -> T2 -> T3-raw); all-tiers-fail => bracket closed. Air
# walks 16K steps (prior 128-144K, bracket comes free); Flash walks 64K then
# bisects to <=16K. Every probe banks as a normal cell. Ends with ONE 1r
# confirm at each model's cap.
# Env required: OCC_PIDS (occupier pids).
set -u
export GPU="0,1" HOSTFLOOR=500
source /workspace/AsymGEMM-SFT/third_party/AsymGEMM/agent/anchors_tmp/stdtps96_lib.sh
export GPU_POOL="0,1" DDP_TIMEOUT=1500
POL="none|false|false|false|false|false"
SDP="asym_sdp2_cpuadamwds"
echo "== stdtps96 A2 phaseA start $(date +%F_%H:%M) occ='$OCC_PIDS' pair=c12:2+3(same-socket dev.) ==" >> "$S"

try_tiers() { # MODEL SEQ TAGBASE ARMENV [LADDER] -> echoes TRAINED tier or FAIL
  local model="$1" seq="$2" tb="$3" arm="$4" ladder="${5:-T1 T2 T3}" t v
  for t in $ladder; do
    v=$(ARM_ENV="$arm" run_cell "${tb}${t,,}" "$model" "$SDP|$t" "$seq" "1" "$POL" 2)
    [ "$v" = "TRAINED" ] && { echo "$t"; return 0; }
    [ "$v" = "OCCDEAD" ] && { echo OCCDEAD; return 1; }
    # non-OOM FAIL: don't ladder past an infra failure
    [ "$v" = "FAIL" ] && { echo FAIL; return 1; }
  done
  echo DEAD; return 0
}

# ---- GLM-4.5-Air: prior 128-144K, 16K walk. ctx 131k; beyond-ctx legal.
AIRARM="ASYM_ARENA_SHM_CAP_GB=400"
cap_air=${AIR_RESUME_CAP:-0}; s=${AIR_RESUME_S:-128}
AIR_DONE=${AIR_DONE:-0}
LAD="${AIR_LADDER:-T1 T2 T3}"
# ensure a floor: walk down if even the prior start is dead
if [ "$AIR_DONE" = 1 ]; then r=SKIPPED; echo "PHASEA air SKIP (done, cap=${cap_air}K)" >> "$S"
elif [ "$cap_air" -gt 0 ]; then r=RESUMED; else
r=$(try_tiers glm4.5-air $((s*1000)) "s96air2c${s}_" "$AIRARM" "$LAD") || exit 1
fi
if [ "$r" = "DEAD" ]; then
  while [ $s -gt 32 ]; do
    s=$((s-16))
    r=$(try_tiers glm4.5-air $((s*1000)) "s96air2c${s}_" "$AIRARM" "$LAD") || exit 1
    [ "$r" != "DEAD" ] && break
  done
fi
if [ "$r" != "DEAD" ] && [ "$r" != "SKIPPED" ]; then
  [ "$r" != "RESUMED" ] && cap_air=$s
  while :; do
    n=$((s+16))
    r=$(try_tiers glm4.5-air $((n*1000)) "s96air2c${n}_" "$AIRARM" "$LAD") || exit 1
    [ "$r" = "DEAD" ] && break
    cap_air=$n; s=$n
    [ $n -ge 320 ] && break   # sanity ceiling: 185G 2r max was 320K
  done
fi
echo "PHASEA air 2r cap=${cap_air}K $(date +%H:%M)" >> "$S"

# ---- GLM-4.7-Flash: prior 448-512K, 64K walk then 16K bisect.
FLARM="ASYM_ARENA_SHM_CAP_GB=400"
cap_fl=0; lo=${FL_RESUME_LO:-0}; hi=0; s=${FL_RESUME_LO:-448}
FLLAD="${FL_LADDER:-T1 T2 T3}"
if [ "$lo" -gt 0 ]; then r=RESUMED; else
r=$(try_tiers glm4.7-flash $((s*1000)) "s96fl2c${s}_" "$FLARM" "$FLLAD") || exit 1
fi
if [ "$r" = "DEAD" ]; then
  while [ $s -gt 64 ]; do
    s=$((s-64))
    r=$(try_tiers glm4.7-flash $((s*1000)) "s96fl2c${s}_" "$FLARM") || exit 1
    [ "$r" != "DEAD" ] && break
  done
fi
if [ "$r" != "DEAD" ]; then
  [ "$r" != "RESUMED" ] && lo=$s
  while :; do
    n=$((s+64))
    r=$(try_tiers glm4.7-flash $((n*1000)) "s96fl2c${n}_" "$FLARM" "$FLLAD") || exit 1
    if [ "$r" = "DEAD" ]; then hi=$n; break; fi
    lo=$n; s=$n
    [ $n -ge 1024 ] && { hi=0; break; }
  done
  # bisect (lo trained, hi dead) to <=16K
  while [ $hi -gt 0 ] && [ $((hi-lo)) -gt 16 ]; do
    m=$(( (lo+hi)/2 )); m=$(( m/16*16 ))
    [ $m -le $lo ] && m=$((lo+16))
    r=$(try_tiers glm4.7-flash $((m*1000)) "s96fl2c${m}_" "$FLARM" "$FLLAD") || exit 1
    if [ "$r" = "DEAD" ]; then hi=$m; else lo=$m; fi
  done
  cap_fl=$lo
fi
echo "PHASEA flash 2r cap=${cap_fl}K $(date +%H:%M)" >> "$S"

# ---- ONE 1r confirm at each cap (inside GPU 0 only)
export GPU=0 CUDA_VISIBLE_DEVICES=0; unset GPU_POOL DDP_TIMEOUT || true
if [ "$cap_air" -gt 0 ]; then
  v=$(ARM_ENV="$AIRARM" run_cell "s96air1c${cap_air}" glm4.5-air "asym_cpuadamwds|T1" $((cap_air*1000)) "1" "$POL" 1)
  [ "$v" != "TRAINED" ] && ARM_ENV="$AIRARM" run_cell "s96air1c${cap_air}b" glm4.5-air "asym_cpuadamwds|T2" $((cap_air*1000)) "1" "$POL" 1
fi
if [ "$cap_fl" -gt 0 ]; then
  v=$(ARM_ENV="" run_cell "s96fl1c${cap_fl}" glm4.7-flash "asym_cpuadamwds|T1" $((cap_fl*1000)) "1" "$POL" 1)
  [ "$v" != "TRAINED" ] && ARM_ENV="" run_cell "s96fl1c${cap_fl}b" glm4.7-flash "asym_cpuadamwds|T2" $((cap_fl*1000)) "1" "$POL" 1
fi
echo "== stdtps96 A2 phaseA DONE air=${cap_air}K flash=${cap_fl}K $(date +%F_%H:%M) ==" >> "$S"
