#!/bin/bash
# STDTPS Agent-3 phase 1: GLM-4.5-Air 2-rank CAP GATE (handoff): probe
# asym-2r @448k AND @384k (grid A 128-448k needs both rungs; fallback grid B
# 64-384k needs 384k). Tier ladder per GLM precedent (glmext_rev):
# sdp2-T1 -> T2 -> T3-raw, b1, shm_guard before every fabric cell.
# GPUs 0+1 (GPU1 ~180 GB free — note edge GOOMs as inconclusive).
set -uo pipefail
export GPU="0,1" HOSTFLOOR=1200
source /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/stdtps_lib.sh
export CUDA_VISIBLE_DEVICES="0,1" GPU_POOL="0,1" DDP_TIMEOUT=1500
export ASYM_ARENA_SHM_CAP_GB=240   # Air banks ~200 GB (GLMTP precedent)
POL="none|false|false|false|false|false"
T3TOK="asym_sdp2_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0"

cell() { local v; v=$(run_cell "$1" glm4.5-air "$2" "$3" "$4" "$POL" 2)
  echo "STDTPS-A3 CELL $1 r2 ${2%%|*} s=$3 -> $v $(date +%H:%M:%S)" >> "$S"; echo "$v"; }

ladder() { local seq="$1" sk=$(( $1 / 1000 )) v
  shm_guard
  v=$(cell "st3air2t1${sk}" "asym_sdp2_cpuadamwds|T1" "$seq" "1")
  [ "$v" = "TRAINED" ] && { echo "T1"; return 0; }
  shm_guard
  v=$(cell "st3air2t2${sk}" "asym_sdp2_cpuadamwds|T2" "$seq" "1")
  [ "$v" = "TRAINED" ] && { echo "T2"; return 0; }
  shm_guard
  v=$(cell "st3air2t3${sk}" "$T3TOK" "$seq" "1")
  [ "$v" = "TRAINED" ] && { echo "T3"; return 0; }
  echo "DEAD"; return 1; }

echo "STDTPS-A3-GATE BEGIN $(date '+%F %H:%M:%S')" >> "$S"
r448=$(ladder 448000) || true
r384=$(ladder 384000) || true
echo "STDTPS-A3-GATE-DONE 448k=$r448 384k=$r384 $(date '+%F %H:%M:%S')" >> "$S"
