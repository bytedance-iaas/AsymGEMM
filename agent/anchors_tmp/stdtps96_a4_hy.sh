#!/bin/bash
# stdtps96_a4_hy.sh — 96G campaign, Agent 4, Hunyuan-A13B **Phase A**:
# 2-rank ceiling search under the HBM occupier (standardize_tps_96gb.md).
# Ladder (hunyuan legal): sdp2 T1 -> T2B (arena 320) -> T3 (arena 320,
# ker101 enabled for this family). Start at the prior 224K, step 32K while
# training, bisect the failing gap to <=16K, then ONE 1r confirm at the cap.
# Every probe is a banked cell (tags s96h2<tier><seqK>).
set -uo pipefail
. /workspace/AsymGEMM-SFT-46/third_party/AsymGEMM/agent/anchors_tmp/stdtps96_lib.sh
POL="none|false|false|false|false|false"
M=hunyuan-a13b
TIER_SPEC=("asym_sdp2_cpuadamwds|T1" "asym_sdp2_cpuadamwds|T2B" "asym_sdp2_cpuadamwds|T3")
# EVERY hunyuan asym cell needs the FAMILY offload list (HY_CAMPAIGN
# integration delta #1: tied embed/lm_head — the driver-default "all" takes
# the untie-by-cloning path, which SEGFAULTS downstream in the attn dense
# LoRA-A cpu-left kernel; every banked hy cell across c17/c11/c18 used the
# exclusion list. FG-PROBE1-8 record, STDTPS96_c18.md incident 4.)
HYOFF="ASYM_OFFLOAD_MODULES=routed_experts,shared_experts,attention,norms,mlp_dense"
TIER_ARM=("$HYOFF" "$HYOFF ASYM_ARENA_SHM_CAP_GB=320" "$HYOFF ASYM_ARENA_SHM_CAP_GB=320")
TIER_TAG=(t1 t2b t3)
start_occupiers || exit 1
echo "=== STDTPS96-A4-HY PHASE-A BEGIN $(date '+%F %H:%M:%S') ===" >> "$S"

LAST_TI=0
# try_rung SEQK START_TI RANKS -> sets TRY_TI; returns 0 on a TRAINED tier
try_rung() { local sk=$1 ti=$2 ranks=${3:-2} v pre
  [ "$ranks" = "1" ] && pre="s96h1" || pre="s96h2"
  for ((; ti<${#TIER_SPEC[@]}; ti++)); do
    v=$(ARM_ENV="${TIER_ARM[$ti]}" run_cell "${pre}${TIER_TAG[$ti]}${sk}" $M "${TIER_SPEC[$ti]}" $((sk*1000)) "1" "$POL" "$ranks")
    echo "A4-HY r${ranks} ${TIER_TAG[$ti]}@${sk}k -> $v" >> "$S"
    if [ "$v" = "TRAINED" ]; then TRY_TI=$ti; return 0; fi
    [ "$v" = "FAIL" ] && { echo "A4-HY HARDFAIL ${TIER_TAG[$ti]}@${sk}k — inspect" >> "$S"; }
  done
  return 1
}

CAP=0; FIRSTFAIL=0
if try_rung 224 0 2; then
  CAP=224; LAST_TI=$TRY_TI
  cur=256
  while try_rung $cur $LAST_TI 2; do CAP=$cur; LAST_TI=$TRY_TI; cur=$((cur+32)); done
  FIRSTFAIL=$cur
else
  FIRSTFAIL=224; cur=192
  while [ $cur -ge 64 ]; do
    if try_rung $cur 0 2; then CAP=$cur; LAST_TI=$TRY_TI; break; fi
    FIRSTFAIL=$cur; cur=$((cur-32))
  done
fi
# bisect (16K granularity)
while [ $CAP -gt 0 ] && [ $((FIRSTFAIL-CAP)) -gt 16 ]; do
  mid=$(( (CAP+FIRSTFAIL)/2/16*16 ))
  [ $mid -le $CAP ] && break
  if try_rung $mid $LAST_TI 2; then CAP=$mid; LAST_TI=$TRY_TI; else FIRSTFAIL=$mid; fi
done
echo "A4-HY 2R CAP=${CAP}K (tier ${TIER_TAG[$LAST_TI]}) wall bracket (${CAP}K,${FIRSTFAIL}K] $(date +%H:%M)" >> "$S"
# ONE 1r confirm at the cap
if [ $CAP -gt 0 ]; then
  if try_rung $CAP $LAST_TI 1; then
    echo "A4-HY 1R CONFIRM @${CAP}K TRAINED (tier ${TIER_TAG[$TRY_TI]}) — cap stands $(date +%H:%M)" >> "$S"
  else
    echo "A4-HY 1R CONFIRM @${CAP}K FAILED — ANOMALY, cap must shrink to the 1r max (re-derive) $(date +%H:%M)" >> "$S"
  fi
fi
echo "=== STDTPS96-A4-HY PHASE-A DONE CAP=${CAP}K $(date '+%F %H:%M:%S') ===" >> "$S"
