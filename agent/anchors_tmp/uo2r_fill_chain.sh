#!/bin/bash
# 2-rank unsloth-off (SuperOffload + Unsloth-GC-Offload) fill campaign, c17:
# panels showing the series in the legend but missing the bars.
# Order (user 2026-08-05): MoE q3-30b-a3b first, then dense q3-32b, llama3.3-70b.
# House protocol via tpfig_lib_c17.sh (w1+m2, batch walk, guard, jobs.tsv verdicts).
set -uo pipefail
export GPU="0,1" HOSTFLOOR=1300
source /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib_c17.sh
export GPU_POOL="0,1"
S2="/workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/uo2r_status.log"
UO="superoffload_mem|unsloth-off-ohbm0"
POL="none|false|false|false|false|false"

cell() { # $1 tag $2 model $3 seq $4 blist -> verdict; logs to both status files
  local v
  v=$(run_cell "$1" "$2" "$UO" "$3" "$4" "$POL" 2)
  echo "UO2R CELL $1 $2 s=$3 -> $v $(date +%H:%M:%S)" >> "$S2"
  echo "$v"
}

# ── MoE: q3-30b-a3b, panel rungs 384k..1.04M, climb until wall ──
v=$(cell uo30s384 q3-30b-a3b 384000 "2 1")
if [ "$v" = "TRAINED" ]; then
  for s in 640000 720000 800000 880000 960000 1040000; do
    v=$(cell "uo30s$((s/1000))" q3-30b-a3b "$s" "1")
    [ "$v" = "TRAINED" ] || { echo "UO2R 30b wall at $s ($v)" >> "$S2"; break; }
  done
else
  echo "UO2R 30b wall at 384000 ($v)" >> "$S2"
fi

# ── dense: q3-32b, rungs 128k..416k ──
for spec in "128000|4 2" "168000|2 1" "256000|2 1" "320000|1" "384000|1" "416000|1"; do
  s="${spec%%|*}"; bl="${spec#*|}"
  v=$(cell "uo32s$((s/1000))" q3-32b "$s" "$bl")
  [ "$v" = "TRAINED" ] || { echo "UO2R 32b wall at $s ($v)" >> "$S2"; break; }
done

# ── dense: llama3.3-70b, rungs 104k..256k ──
for spec in "104000|4 2" "128000|2 1" "168000|2 1" "192000|1" "224000|1" "256000|1"; do
  s="${spec%%|*}"; bl="${spec#*|}"
  v=$(cell "uo70s$((s/1000))" llama3.3-70b "$s" "$bl")
  [ "$v" = "TRAINED" ] || { echo "UO2R 70b wall at $s ($v)" >> "$S2"; break; }
done

echo "UO2R_DONE $(date +%H:%M:%S)" >> "$S2"
