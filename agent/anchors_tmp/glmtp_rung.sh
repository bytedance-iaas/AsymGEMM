#!/bin/bash
# One full rung for the GLM tp campaign: rc -> uns -> uns_off -> asym tiers.
# Usage: glmtp_rung.sh PHASE MODEL SEQ "BLIST" RANKS
# PHASE = tag prefix (f1/f2/a1/a2). Asym: T1 walk; on no-fit -> T2 top-2
# batches; on no-fit -> T3-raw top-2. Emits RUNG-DONE marker per rung.
set -uo pipefail
PHASE="$1"; MODEL="$2"; SEQ="$3"; BLIST="$4"; RANKS="${5:-1}"
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
# asym_cpuadamwds is single-GPU; |2 rows use the paper's 2-rank asym
# system: shared-fabric streaming DP (ep2-vanilla's swap is qwen3-only)
ASYM_BE=asym_cpuadamwds; [ "$RANKS" = "2" ] && ASYM_BE=asym_sdp2_cpuadamwds
T3TOK="${ASYM_BE}|recomp-off-full-fg-ker000-ceil0000-ohbm0"
sk=$((SEQ/1000))

run_cell "${PHASE}rc${sk}" "$MODEL" "superoffload_mem|recomp"          "$SEQ" "$BLIST" "none|false|false|false|false|false" "$RANKS"
run_cell "${PHASE}un${sk}" "$MODEL" "superoffload_mem|unsloth"         "$SEQ" "$BLIST" "none|false|false|false|false|false" "$RANKS"
run_cell "${PHASE}uo${sk}" "$MODEL" "superoffload_mem|unsloth-off-ohbm0" "$SEQ" "$BLIST" "none|false|false|false|false|false" "$RANKS"
v=$(run_cell "${PHASE}t1${sk}" "$MODEL" "${ASYM_BE}|T1" "$SEQ" "$BLIST" "none|false|false|false|false|false" "$RANKS")
if [ "$v" != "TRAINED" ]; then
  top2=$(echo $BLIST | awk '{print $1, $2}')
  v=$(run_cell "${PHASE}t2${sk}" "$MODEL" "${ASYM_BE}|T2" "$SEQ" "$top2" "none|false|false|false|false|false" "$RANKS")
  if [ "$v" != "TRAINED" ]; then
    run_cell "${PHASE}t3${sk}" "$MODEL" "$T3TOK" "$SEQ" "$top2" "none|false|false|false|false|false" "$RANKS"
  fi
fi
echo "RUNG-DONE ${PHASE} ${MODEL} s=${SEQ} $(date +%H:%M)" >> "$S"
