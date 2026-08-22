#!/bin/bash
# STDTPS96 Agent-1 Phase C (part 2): 30B 1-RANK grid fill on the SAME rungs
# 128/256/384/512/640K (+768K asym = banked cap1r confirm; uo@640/768K
# conditional). Phys GPU0 (inside idx 0). Ladder T1->T2->T2B->T3 forward-only.
set -uo pipefail
export GPU=0 HOSTFLOOR=1100
source /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/stdtps96_lib.sh
export CUDA_VISIBLE_DEVICES=0
unset GPU_POOL DDP_TIMEOUT || true
POL="none|false|false|false|false|false"
RC="superoffload_mem|recomp"; UN="superoffload_mem|unsloth"
UO="superoffload_mem|unsloth-off-ohbm0"; Z3="zero3_offload_mem|recomp"; FD="fsdp2_offload|recomp"
TIERS=(T1 T2 T2B T3); ti=0
declare -A DEAD=()

cell() { local v; shm_guard
  v=$(run_cell "$1" q3-30b-a3b "$2" "$3" "$4" "$POL" 1)
  echo "STDTPS96-A1 CELL $1 r1 ${2#*|} s=$3 -> $v $(date +%H:%M:%S)" >> "$S"; echo "$v"; }

base() { local key=$1 tok=$2 seq=$3 blist=$4 v
  [ "${DEAD[$key]:-0}" = "1" ] && return 0
  v=$(cell "s96q30f1${key}_$((seq/1000))" "$tok" "$seq" "$blist")
  case "$v" in
    TRAINED) ;;
    GOOM|COOM) DEAD[$key]=1; echo "STDTPS96-A1 r1 $key WALL at $seq ($v)" >> "$S";;
    *) echo "STDTPS96-A1 FAIL-STOP $key s=$seq v='$v'" >> "$S"; exit 1;;
  esac; }

asym() { local seq=$1 t v
  while [ $ti -lt ${#TIERS[@]} ]; do
    t=${TIERS[$ti]}
    v=$(cell "s96q30f1as_$((seq/1000))${t,,}" "asym_cpuadamwds|$t" "$seq" "$2")
    case "$v" in
      TRAINED) return 0;;
      GOOM|COOM) ti=$((ti+1));;
      *) echo "STDTPS96-A1 FAIL-STOP asym s=$seq v='$v'" >> "$S"; exit 1;;
    esac
  done
  echo "STDTPS96-A1 r1 ASYM DEAD at $seq (below the 1r-confirmed cap!)" >> "$S"; exit 1; }

echo "STDTPS96-A1-FILL1R BEGIN $(date '+%F %H:%M:%S')" >> "$S"
base rc "$RC" 128000 "2 1"; base un "$UN" 128000 "2 1"; base uo "$UO" 128000 "4 2 1"
base z3 "$Z3" 128000 "2 1"; base fd "$FD" 128000 "2 1"; asym 128000 "2 1"
base rc "$RC" 256000 "1";   base un "$UN" 256000 "1";   base uo "$UO" 256000 "2 1"
base z3 "$Z3" 256000 "1";   base fd "$FD" 256000 "1";   asym 256000 "1"
base rc "$RC" 384000 "1";   base un "$UN" 384000 "1";   base uo "$UO" 384000 "1"
base z3 "$Z3" 384000 "1";   base fd "$FD" 384000 "1";   asym 384000 "1"
base rc "$RC" 512000 "1";   base un "$UN" 512000 "1";   base uo "$UO" 512000 "1"
base z3 "$Z3" 512000 "1";   base fd "$FD" 512000 "1";   asym 512000 "1"
base rc "$RC" 640000 "1";   base un "$UN" 640000 "1";   base uo "$UO" 640000 "1"
base z3 "$Z3" 640000 "1";   base fd "$FD" 640000 "1";   asym 640000 "1"
# 768K: asym = banked cap1r cell; uo probe only if still alive
base uo "$UO" 768000 "1"
echo "STDTPS96-A1-FILL1R-DONE $(date '+%F %H:%M:%S')" >> "$S"
