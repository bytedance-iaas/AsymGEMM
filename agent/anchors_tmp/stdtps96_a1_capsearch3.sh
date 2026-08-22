#!/bin/bash
# STDTPS96 Agent-1 Phase A v3 — fixes v2's subshell bug (tier demotions in
# attempt() didn't persist: ATT/ti are now globals, attempt called directly).
# RESUME STATE (banked): 896K T2/T2B/T3 all GOOM (T3 was the clean rerun);
# 768K T2/T2B GOOM, T3 TRAINED  =>  ok=768K(T3), f=896K, ti=2 (T3).
# Remaining: bisect (768K,896K] to <=16K with T3, then the 1r confirm.
set -uo pipefail
export GPU="0,1" HOSTFLOOR=1100
source /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/stdtps96_lib.sh
export CUDA_VISIBLE_DEVICES="0,1" GPU_POOL="0,1"
POL="none|false|false|false|false|false"
SEP=asym_sepplan2_cpuadamwds
TIERS=(T2 T2B T3)
ti=2
ATT=""

cell() { local v; shm_guard
  v=$(run_cell "$1" q3-30b-a3b "$2" "$3" "1" "$POL" "$4")
  echo "STDTPS96-A1 CELL $1 r$4 ${2#*|} s=$3 -> $v $(date +%H:%M:%S)" >> "$S"; echo "$v"; }

attempt() { local seq=$1 t v   # sets ATT + advances global ti on OOM
  ATT=DEAD
  while [ $ti -lt ${#TIERS[@]} ]; do
    t=${TIERS[$ti]}
    v=$(cell "s96q30${t,,}_$((seq/1000))" "$SEP|$t" "$seq" 2)
    case "$v" in
      TRAINED)   ATT=TRAINED; return 0;;
      GOOM|COOM) ti=$((ti+1));;
      *)         echo "STDTPS96-A1 FAIL-STOP verdict='$v' s=$seq $(date +%H:%M:%S)" >> "$S"
                 ATT=FAIL; return 0;;
    esac
  done; }

echo "STDTPS96-A1-CAPSEARCH-V3 BEGIN (bisect (768K,896K] @T3) $(date '+%F %H:%M:%S')" >> "$S"
ok=768000; f=896000
while [ $((f-ok)) -gt 16000 ]; do
  m=$(( (ok+f)/2 )); m=$(( m/16000*16000 ))
  [ $m -le $ok ] && m=$((ok+16000)); [ $m -ge $f ] && m=$((f-16000))
  attempt $m
  [ "$ATT" = FAIL ] && exit 1
  if [ "$ATT" = TRAINED ]; then ok=$m; else f=$m; fi
done
CT=${TIERS[$ti]:-T3}
echo "STDTPS96-A1-CAP ok=$ok bracket=($ok,$f] tier=$CT $(date '+%F %H:%M:%S')" >> "$S"
export GPU=0 CUDA_VISIBLE_DEVICES=0; unset GPU_POOL || true
v=$(cell "s96q30cap1r_$((ok/1000))" "asym_cpuadamwds|$CT" "$ok" 1)
echo "STDTPS96-A1-CAPSEARCH-DONE cap=$ok tier=$CT confirm1r=$v $(date '+%F %H:%M:%S')" >> "$S"
