#!/bin/bash
# STDTPS96 Agent-4 sweep / Hunyuan Phase A: 2r ceiling search under the 96G
# occupiers. Prior ~224-256K -> start 256K. Hunyuan legal ladder T1->T2B->T3
# (HY_CAMPAIGN; T2B/T3 need ASYM_ARENA_SHM_CAP_GB=320). sdp2 backend, walk
# +-32K (parent step), bisect <=16K, then 1r confirm at cap.
set -uo pipefail
export GPU="0,1" HOSTFLOOR=1100
source /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/stdtps96_lib.sh
export CUDA_VISIBLE_DEVICES="0,1" GPU_POOL="0,1"
export ASYM_OFFLOAD_MODULES="routed_experts,shared_experts,attention,norms,mlp_dense"
POL="none|false|false|false|false|false"
SDP=asym_sdp2_cpuadamwds
T3TOK="$SDP|recomp-off-full-fg-ker101-ceil0000-ohbm0"   # hunyuan T3 = route-kernel (ker101 legal since T3-enablement)
TIERS=(T1 T2B T3); ti=0; ATT=""

cell() { local v; shm_guard
  v=$(run_cell "$1" hunyuan-a13b "$2" "$3" "1" "$POL" "$4")
  echo "STDTPS96-A4 CELL $1 r$4 ${2#*|} s=$3 -> $v $(date +%H:%M:%S)" >> "$S"; echo "$v"; }

attempt() { local seq=$1 t tok v
  ATT=DEAD
  while [ $ti -lt ${#TIERS[@]} ]; do
    t=${TIERS[$ti]}
    tok="$SDP|$t"; [ "$t" = "T3" ] && tok="$T3TOK"
    if [ "$t" != "T1" ]; then export ASYM_ARENA_SHM_CAP_GB=320; else unset ASYM_ARENA_SHM_CAP_GB || true; fi
    v=$(cell "s96hy${t,,}_$((seq/1000))" "$tok" "$seq" 2)
    case "$v" in
      TRAINED)   ATT=TRAINED; return 0;;
      GOOM|COOM) ti=$((ti+1));;
      *) echo "STDTPS96-A4 FAIL-STOP verdict='$v' s=$seq $(date +%H:%M:%S)" >> "$S"; ATT=FAIL; return 0;;
    esac
  done; }

echo "STDTPS96-A4-HYCAP BEGIN $(date '+%F %H:%M:%S')" >> "$S"
ok=0; f=0; s=256000
while :; do
  attempt $s
  [ "$ATT" = FAIL ] && exit 1
  if [ "$ATT" = TRAINED ]; then ok=$s; break; fi
  f=$s; ti=0
  s=$((s-32000))
  [ $s -lt 64000 ] && { echo "STDTPS96-A4 HY ALL-DEAD $(date +%H:%M:%S)" >> "$S"; exit 1; }
done
if [ $f -eq 0 ]; then
  s=$((ok+32000))
  while [ $s -le 512000 ]; do
    attempt $s
    [ "$ATT" = FAIL ] && exit 1
    if [ "$ATT" = TRAINED ]; then ok=$s; s=$((s+32000)); else f=$s; break; fi
  done
  [ $f -eq 0 ] && f=$((ok+32000))
fi
while [ $((f-ok)) -gt 16000 ]; do
  m=$(( (ok+f)/2 )); m=$(( m/16000*16000 ))
  [ $m -le $ok ] && m=$((ok+16000)); [ $m -ge $f ] && m=$((f-16000))
  attempt $m
  [ "$ATT" = FAIL ] && exit 1
  if [ "$ATT" = TRAINED ]; then ok=$m; else f=$m; fi
done
CT=${TIERS[$ti]:-T3}
echo "STDTPS96-A4-HYCAP ok=$ok bracket=($ok,$f] tier=$CT $(date '+%F %H:%M:%S')" >> "$S"
export GPU=0 CUDA_VISIBLE_DEVICES=0; unset GPU_POOL || true
T1RTOK="asym_cpuadamwds|$CT"; [ "$CT" = "T3" ] && T1RTOK="asym_cpuadamwds|recomp-off-full-fg-ker101-ceil0000-ohbm0"
v=$(cell "s96hycap1r_$((ok/1000))" "$T1RTOK" "$ok" 1)
echo "STDTPS96-A4-HYCAP-DONE cap=$ok tier=$CT confirm1r=$v $(date '+%F %H:%M:%S')" >> "$S"
