#!/bin/bash
# HY 1r resume (rungs 160-256K) after the dataset_info race FAIL-STOP.
# State: rc DEAD (wall (96K,128K]); un/uo alive; asym ladder at T1.
set -uo pipefail
export GPU=0 HOSTFLOOR=1100
source /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/stdtps96_lib.sh
export CUDA_VISIBLE_DEVICES=0
unset GPU_POOL DDP_TIMEOUT ASYM_ARENA_SHM_CAP_GB || true
export ASYM_OFFLOAD_MODULES="routed_experts,shared_experts,attention,norms,mlp_dense"
POL="none|false|false|false|false|false"
UN="superoffload_mem|unsloth-ohbm0"; UO="superoffload_mem|unsloth-off-ohbm0"
T3TOK="asym_cpuadamwds|recomp-off-full-fg-ker101-ceil0000-ohbm0"
TIERS=(T1 T2B T3); ti=0
declare -A DEAD=()

cell() { local v; shm_guard
  v=$(run_cell "$1" hunyuan-a13b "$2" "$3" "$4" "$POL" 1)
  echo "STDTPS96-A4 CELL $1 r1 ${2#*|} s=$3 -> $v $(date +%H:%M:%S)" >> "$S"; echo "$v"; }
base() { local key=$1 tok=$2 seq=$3 blist=$4 v
  [ "${DEAD[$key]:-0}" = "1" ] && return 0
  v=$(cell "s96hyf1${key}_$((seq/1000))b" "$tok" "$seq" "$blist")
  case "$v" in
    TRAINED) ;;
    GOOM|COOM) DEAD[$key]=1; echo "STDTPS96-A4 r1 $key WALL at $seq ($v)" >> "$S";;
    *) echo "STDTPS96-A4 FAIL-STOP $key s=$seq v='$v'" >> "$S"; exit 1;;
  esac; }
asym() { local seq=$1 t tok v
  while [ $ti -lt ${#TIERS[@]} ]; do
    t=${TIERS[$ti]}
    tok="asym_cpuadamwds|$t"; [ "$t" = "T3" ] && tok="$T3TOK"
    v=$(cell "s96hyf1as_$((seq/1000))${t,,}b" "$tok" "$seq" "$2")
    case "$v" in
      TRAINED) return 0;;
      GOOM|COOM) ti=$((ti+1));;
      *) echo "STDTPS96-A4 FAIL-STOP asym s=$seq v='$v'" >> "$S"; exit 1;;
    esac
  done
  echo "STDTPS96-A4 r1 ASYM DEAD at $seq" >> "$S"; exit 1; }

echo "STDTPS96-A4-HY1R-B BEGIN (resume 160K) $(date '+%F %H:%M:%S')" >> "$S"
base un "$UN" 160000 "1"; base uo "$UO" 160000 "2 1"; asym 160000 "1"
base un "$UN" 192000 "1"; base uo "$UO" 192000 "1";   asym 192000 "1"
base un "$UN" 224000 "1"; base uo "$UO" 224000 "1";   asym 224000 "1"
base un "$UN" 256000 "1"; base uo "$UO" 256000 "1";   asym 256000 "1"
echo "STDTPS96-A4-HY1R-B-DONE $(date '+%F %H:%M:%S')" >> "$S"
