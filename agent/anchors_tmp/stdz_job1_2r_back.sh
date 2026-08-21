#!/bin/bash
# stdz_job1_2r_back.sh — Kevin-directed work-steal: c12 chews the 30B-2r
# block from the BACK (1.02M -> 896k -> 768k) while c18 goes forward from
# 512k. Backend/recipe = c18's banner (asym_sepplan2_cpuadamwds|T2, GLOBAL
# tok/s, w1+m2); tags s2b30*-c12; cross-machine waiver (row is c18-native)
# noted in the DATA comment at bank time. YIELD RULE: the pre-cell guard
# greps the doc for a c18 "STARTED/PROGRESS" line naming the cell's seq and
# skips it if found (both sides post before starting).
set -u
export GPU="0,1" HOSTFLOOR=1100
source /workspace/AsymGEMM-SFT/third_party/AsymGEMM/agent/anchors_tmp/fig12_lib.sh
export GPU_POOL="0,1" DDP_TIMEOUT=1500
S="$LOGD/stdz_status.log"
DOC=/workspace/AsymGEMM-SFT/third_party/AsymGEMM/agent/impls/s04-p1-dgx-02-c06/standardize_tps.md
POL="none|false|false|false|false|false"
SEP="asym_sepplan2_cpuadamwds|T2"
echo "== stdz job1 2r-back (work-steal) start $(date +%F_%H:%M) ==" >> "$S"
taken_by_c18() { grep -qiE "c18.*(STARTED|PROGRESS|BANKED).*$1|$1.*(STARTED|BANKED).*c18" "$DOC"; }
for sq in 1024000 896000 768000; do
  lbl=$sq; [ "$sq" = 1024000 ] && lbl="1\\.02M|1024k"
  if taken_by_c18 "$lbl"; then
    echo "CELL s2b30_${sq} -> YIELD (c18 owns per doc) $(date +%H:%M)" >> "$S"
    continue
  fi
  echo "STARTED s2b30_${sq} on c12 (work-steal back-end) $(date +%F_%H:%M)" >> "$S"
  ARM_ENV="" run_cell "s2b30_$((sq/1000))" q3-30b-a3b "$SEP" "$sq" "1" "$POL" 2
done
echo "== stdz job1 2r-back DONE $(date +%F_%H:%M) ==" >> "$S"
