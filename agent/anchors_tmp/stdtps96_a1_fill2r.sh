#!/bin/bash
# STDTPS96 Agent-1 Phase C (part 1): 30B 2-RANK grid fill, rungs 128/256/
# 384/512/640K (768K = the banked Phase-A T3 crown, skipped). Low->high;
# per-system monotone: first b1 G/C-OOM kills the system for higher rungs
# (banked OOM by monotonicity). Asym ladder T2->T2B->T3, forward-only tier
# index (an OOM at rung r implies OOM above). shm_guard in cell();
# guard auto-cleans stale asym shm (lib).
set -uo pipefail
export GPU="0,1" HOSTFLOOR=1100
source /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/stdtps96_lib.sh
export CUDA_VISIBLE_DEVICES="0,1" GPU_POOL="0,1"
POL="none|false|false|false|false|false"
SEP=asym_sepplan2_cpuadamwds
RC="superoffload_mem|recomp"; UN="superoffload_mem|unsloth"
UO="superoffload_mem|unsloth-off-ohbm0"; Z3="zero3_offload_mem|recomp"; FD="fsdp2_offload|recomp"
TIERS=(T2 T2B T3); ti=0
declare -A DEAD=()

cell() { local v; shm_guard
  v=$(run_cell "$1" q3-30b-a3b "$2" "$3" "$4" "$POL" 2)
  echo "STDTPS96-A1 CELL $1 r2 ${2#*|} s=$3 -> $v $(date +%H:%M:%S)" >> "$S"; echo "$v"; }

base() { local key=$1 tok=$2 seq=$3 blist=$4 v
  [ "${DEAD[$key]:-0}" = "1" ] && return 0
  v=$(cell "s96q30f2${key}_$((seq/1000))" "$tok" "$seq" "$blist")
  case "$v" in
    TRAINED) ;;
    GOOM|COOM) DEAD[$key]=1; echo "STDTPS96-A1 r2 $key WALL at $seq ($v)" >> "$S";;
    *) echo "STDTPS96-A1 FAIL-STOP $key s=$seq v='$v'" >> "$S"; exit 1;;
  esac; }

asym() { local seq=$1 t v
  while [ $ti -lt ${#TIERS[@]} ]; do
    t=${TIERS[$ti]}
    v=$(cell "s96q30f2as_$((seq/1000))${t,,}" "$SEP|$t" "$seq" "$2")
    case "$v" in
      TRAINED) return 0;;
      GOOM|COOM) ti=$((ti+1));;
      *) echo "STDTPS96-A1 FAIL-STOP asym s=$seq v='$v'" >> "$S"; exit 1;;
    esac
  done
  echo "STDTPS96-A1 r2 ASYM DEAD at $seq (unexpected below cap!)" >> "$S"; exit 1; }

echo "STDTPS96-A1-FILL2R BEGIN $(date '+%F %H:%M:%S')" >> "$S"
# ---- 128K ----
base rc "$RC" 128000 "1";  base un "$UN" 128000 "2 1"; base uo "$UO" 128000 "2 1"
base z3 "$Z3" 128000 "1";  base fd "$FD" 128000 "1";   asym 128000 "2 1"
# ---- 256K ----
base rc "$RC" 256000 "1";  base un "$UN" 256000 "1";   base uo "$UO" 256000 "1"
base z3 "$Z3" 256000 "1";  base fd "$FD" 256000 "1";   asym 256000 "1"
# ---- 384K ----
base rc "$RC" 384000 "1";  base un "$UN" 384000 "1";   base uo "$UO" 384000 "1"
base z3 "$Z3" 384000 "1";  base fd "$FD" 384000 "1";   asym 384000 "1"
# ---- 512K ----
base rc "$RC" 512000 "1";  base un "$UN" 512000 "1";   base uo "$UO" 512000 "1"
base z3 "$Z3" 512000 "1";  base fd "$FD" 512000 "1";   asym 512000 "1"
# ---- 640K ----
base rc "$RC" 640000 "1";  base un "$UN" 640000 "1";   base uo "$UO" 640000 "1"
base z3 "$Z3" 640000 "1";  base fd "$FD" 640000 "1";   asym 640000 "1"
# 768K asym = banked Phase-A crown (s96q30t3_768r); baselines OOM by monotonicity
echo "STDTPS96-A1-FILL2R-DONE $(date '+%F %H:%M:%S')" >> "$S"
