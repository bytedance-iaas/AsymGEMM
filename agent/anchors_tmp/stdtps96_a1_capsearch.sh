#!/bin/bash
# STDTPS96 Agent-1 Phase A: Qwen3-30B-A3B 2-rank CEILING search under the
# 96G occupiers (standardize_tps_96gb.md). Prior ~832-960K -> start 896K.
# sEP backend (banked 30B-2r convention), tier ladder T2->T2B->T3 (T1 dead
# at these depths on 185G; monotone demotion). Walk +-128K, then bisect to
# a <=16K bracket; every probe = banked cell (b1, w1+m2). Ends with the
# 1r confirm at the cap (phys GPU0 = inside idx 0).
# Container GPUs: launcher passes phys "0,3" -> inside indices 0,1.
set -uo pipefail
export GPU="0,1" HOSTFLOOR=1100
source /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/stdtps96_lib.sh
export CUDA_VISIBLE_DEVICES="0,1" GPU_POOL="0,1"
POL="none|false|false|false|false|false"
SEP=asym_sepplan2_cpuadamwds
TIERS=(T2 T2B T3)
ti=0

cell() { local v; v=$(run_cell "$1" q3-30b-a3b "$2" "$3" "1" "$POL" "$4")
  echo "STDTPS96-A1 CELL $1 r$4 ${2#*|} s=$3 -> $v $(date +%H:%M:%S)" >> "$S"; echo "$v"; }

# attempt SEQ -> TRAINED|DEAD|FAIL ; demotes the tier index permanently on OOM
attempt() { local seq=$1 t v
  while [ $ti -lt ${#TIERS[@]} ]; do
    t=${TIERS[$ti]}
    v=$(cell "s96q30${t,,}_$((seq/1000))" "$SEP|$t" "$seq" 2)
    case "$v" in
      TRAINED) echo TRAINED; return 0;;
      FAIL)    echo FAIL; return 0;;
      *)       ti=$((ti+1));;   # G/C-OOM -> leaner tier (monotone)
    esac
  done
  echo DEAD; }

echo "STDTPS96-A1-CAPSEARCH BEGIN $(date '+%F %H:%M:%S')" >> "$S"
ok=0; f=0; s=896000
while :; do
  r=$(attempt $s)
  [ "$r" = FAIL ] && { echo "STDTPS96-A1 FAIL-STOP at s=$s $(date +%H:%M:%S)" >> "$S"; exit 1; }
  if [ "$r" = TRAINED ]; then ok=$s; break; fi
  f=$s; ti=0   # walking DOWN re-arms the full ladder (leaner demotions don't bind below)
  s=$((s-128000))
  [ $s -lt 128000 ] && { echo "STDTPS96-A1 ALL-DEAD (no 2r fit >=128K) $(date +%H:%M:%S)" >> "$S"; exit 1; }
done
if [ $f -eq 0 ]; then
  s=$((ok+128000))
  while [ $s -le 1216000 ]; do
    r=$(attempt $s)
    [ "$r" = FAIL ] && { echo "STDTPS96-A1 FAIL-STOP at s=$s" >> "$S"; exit 1; }
    if [ "$r" = TRAINED ]; then ok=$s; s=$((s+128000)); else f=$s; break; fi
  done
  [ $f -eq 0 ] && f=$((ok+128000))   # sanity-max reached: bracket open above
fi
while [ $((f-ok)) -gt 16000 ]; do
  m=$(( (ok+f)/2 )); m=$(( m/16000*16000 ))
  [ $m -le $ok ] && m=$((ok+16000)); [ $m -ge $f ] && m=$((f-16000))
  r=$(attempt $m)
  [ "$r" = FAIL ] && { echo "STDTPS96-A1 FAIL-STOP at s=$m" >> "$S"; exit 1; }
  if [ "$r" = TRAINED ]; then ok=$m; else f=$m; fi
done
CT=${TIERS[$ti]:-T3}
echo "STDTPS96-A1-CAP ok=$ok bracket=($ok,$f] tier=$CT $(date '+%F %H:%M:%S')" >> "$S"
# 1r confirm at the cap (phys GPU0 = inside index 0)
export GPU=0 CUDA_VISIBLE_DEVICES=0; unset GPU_POOL || true
v=$(cell "s96q30cap1r_$((ok/1000))" "asym_cpuadamwds|$CT" "$ok" 1)
echo "STDTPS96-A1-CAPSEARCH-DONE cap=$ok tier=$CT confirm1r=$v $(date '+%F %H:%M:%S')" >> "$S"
