#!/bin/bash
# STDTPS96 A4-sweep / gpt-oss-20b 1-RANK, METHOD V2 (peak-audit, no occupier):
# rungs 192-576K = the UNION of both candidate grids (2r cap prior 512-576K;
# final axis set when a 2-GPU node can run the 2r cap search). Verdicts from
# exact post-run peak resv (OVER96 = inferred 96G wall, run still TRAINS).
# Ladder T1->T2B->T3-raw (gpt-oss legal ladder), forward-only on OVER96/GOOM.
set -uo pipefail
export GPU=0 HOSTFLOOR=1100
source /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/stdtps96_lib2.sh
unset GPU_POOL DDP_TIMEOUT ASYM_ARENA_SHM_CAP_GB || true
export ASYM_OFFLOAD_MODULES=all
POL="none|false|false|false|false|false"
RC="superoffload_mem|recomp"; UN="superoffload_mem|unsloth-ohbm0"; UO="superoffload_mem|unsloth-off-ohbm0"
T3TOK="asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0"
TIERS=(T1 T2B T3); ti=0
declare -A DEAD=()

cell() { local v
  v=$(run_cell_pa "$1" gpt-oss-20b "$2" "$3" "$4" "$POL" 1)
  echo "STDTPS96-A4 CELL $1 r1 ${2#*|} s=$3 -> $v $(date +%H:%M:%S)" >> "$S"; echo "$v"; }

base() { local key=$1 tok=$2 seq=$3 blist=$4 v
  [ "${DEAD[$key]:-0}" = "1" ] && return 0
  v=$(cell "s96g20f1${key}_$((seq/1000))" "$tok" "$seq" "$blist")
  case "$v" in
    TRAINED_FIT96*|TRAINED_EDGE96*) ;;
    OVER96*|GOOM|COOM) DEAD[$key]=1; echo "STDTPS96-A4 r1 $key 96G-WALL at $seq ($v)" >> "$S";;
    *) echo "STDTPS96-A4 FAIL-STOP $key s=$seq v='$v'" >> "$S"; exit 1;;
  esac; }

asym() { local seq=$1 t tok v
  while [ $ti -lt ${#TIERS[@]} ]; do
    t=${TIERS[$ti]}
    tok="asym_cpuadamwds|$t"; [ "$t" = "T3" ] && tok="$T3TOK"
    v=$(cell "s96g20f1as_$((seq/1000))${t,,}" "$tok" "$seq" "$2")
    case "$v" in
      TRAINED_FIT96*|TRAINED_EDGE96*) return 0;;
      OVER96*|GOOM|COOM) ti=$((ti+1));;
      *) echo "STDTPS96-A4 FAIL-STOP asym s=$seq v='$v'" >> "$S"; exit 1;;
    esac
  done
  echo "STDTPS96-A4 r1 ASYM 96G-DEAD at $seq (1r ceiling found)" >> "$S"; DEAD[asymceil]=1; }

# PREDICTED-AXIS protocol: PHASE 1 = asym ladder only (V2 ceiling scan,
# 64K steps up from 192K until OVER96/ladder-dead); baselines wait for the
# predicted grid (phase-2 chain, written after the cap prediction).
echo "STDTPS96-A4-G20F1 BEGIN (METHOD V2, asym-ladder ceiling scan) $(date '+%F %H:%M:%S')" >> "$S"
for SEQ in 192000 256000 320000 384000 448000 512000 576000 640000; do
  case $SEQ in
    192000) BA="4 2 1";;
    256000) BA="2 1";;
    *)      BA="1";;
  esac
  [ "${DEAD[asymceil]:-0}" = "1" ] && break
  asym $SEQ "$BA"
done
echo "STDTPS96-A4-G20F1-DONE (ceiling scan) $(date '+%F %H:%M:%S')" >> "$S"
