#!/bin/bash
# STDTPS96 Agent-1 Phase A v2 — resumes after the 05:01 shm-leak incident.
# Trusted state: 896K T2 GOOM + T2B GOOM (GPU-side, banked). SUSPECT (ran
# with ~410G leaked fabric shm on the host): 896K T3 COOM, 768K T2 COOM —
# both RE-RUN here on a clean node (lib guard now auto-cleans asym shm).
# Fixes vs v1: only GOOM/COOM demote the tier; ANY other verdict (guard
# fail, FAIL, empty) = FAIL-STOP. shm_guard before every sepplan cell.
set -uo pipefail
export GPU="0,1" HOSTFLOOR=1100
source /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/stdtps96_lib.sh
export CUDA_VISIBLE_DEVICES="0,1" GPU_POOL="0,1"
POL="none|false|false|false|false|false"
SEP=asym_sepplan2_cpuadamwds
TIERS=(T2 T2B T3)
ti=2   # resume: at 896K, T2/T2B already GOOM-banked -> start at T3

cell() { local v; shm_guard
  v=$(run_cell "$1" q3-30b-a3b "$2" "$3" "1" "$POL" "$4")
  echo "STDTPS96-A1 CELL $1 r$4 ${2#*|} s=$3 -> $v $(date +%H:%M:%S)" >> "$S"; echo "$v"; }

attempt() { local seq=$1 t v sfx="${2:-}"
  while [ $ti -lt ${#TIERS[@]} ]; do
    t=${TIERS[$ti]}
    v=$(cell "s96q30${t,,}_$((seq/1000))${sfx}" "$SEP|$t" "$seq" 2)
    case "$v" in
      TRAINED)   echo TRAINED; return 0;;
      GOOM|COOM) ti=$((ti+1));;
      *)         echo "STDTPS96-A1 FAIL-STOP verdict='$v' s=$seq $(date +%H:%M:%S)" >> "$S"
                 echo FAIL; return 0;;
    esac
  done
  echo DEAD; }

echo "STDTPS96-A1-CAPSEARCH-V2 BEGIN (resume 896K@T3 clean) $(date '+%F %H:%M:%S')" >> "$S"
ok=0; f=0; s=896000; sfx="r"   # r = clean-rerun tags (dirs must not collide)
while :; do
  r=$(attempt $s $sfx)
  [ "$r" = FAIL ] && exit 1
  if [ "$r" = TRAINED ]; then ok=$s; break; fi
  f=$s; ti=0; sfx=""
  s=$((s-128000))
  [ $s -lt 128000 ] && { echo "STDTPS96-A1 ALL-DEAD (no 2r fit >=128K) $(date +%H:%M:%S)" >> "$S"; exit 1; }
  [ $s -eq 768000 ] && sfx="r"   # 768K T2 was the other suspect cell
done
if [ $f -eq 0 ]; then
  s=$((ok+128000)); sfx=""
  while [ $s -le 1216000 ]; do
    r=$(attempt $s)
    [ "$r" = FAIL ] && exit 1
    if [ "$r" = TRAINED ]; then ok=$s; s=$((s+128000)); else f=$s; break; fi
  done
  [ $f -eq 0 ] && f=$((ok+128000))
fi
while [ $((f-ok)) -gt 16000 ]; do
  m=$(( (ok+f)/2 )); m=$(( m/16000*16000 ))
  [ $m -le $ok ] && m=$((ok+16000)); [ $m -ge $f ] && m=$((f-16000))
  r=$(attempt $m)
  [ "$r" = FAIL ] && exit 1
  if [ "$r" = TRAINED ]; then ok=$m; else f=$m; fi
done
CT=${TIERS[$ti]:-T3}
echo "STDTPS96-A1-CAP ok=$ok bracket=($ok,$f] tier=$CT $(date '+%F %H:%M:%S')" >> "$S"
export GPU=0 CUDA_VISIBLE_DEVICES=0; unset GPU_POOL || true
v=$(cell "s96q30cap1r_$((ok/1000))" "asym_cpuadamwds|$CT" "$ok" 1)
echo "STDTPS96-A1-CAPSEARCH-DONE cap=$ok tier=$CT confirm1r=$v $(date '+%F %H:%M:%S')" >> "$S"
