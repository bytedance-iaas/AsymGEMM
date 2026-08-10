#!/bin/bash
# One sepplan2 2-rank cell, run INSIDE the container.
# Args: TAG MODEL SYSTOK SEQ BLIST ARENA_GB(0=default) HOSTFLOOR
set -uo pipefail
TAG="$1"; MODEL="$2"; SYSTOK="$3"; SEQ="$4"; BLIST="$5"; ARENA="${6:-0}"; FLOOR="${7:-1100}"
export GPU="0,1" HOSTFLOOR="$FLOOR"
source /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib_c17.sh
export CUDA_VISIBLE_DEVICES="0,1" GPU_POOL="0,1"
S2="$LOGD/sepplan_status.log"
POL="none|false|false|false|false|false"
# stale fabric arenas poison launches — clean when no trainer is alive
if ! nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; then
  rm -f /dev/shm/asym_* 2>/dev/null || true
fi
if [ "$ARENA" != "0" ]; then export ASYM_ARENA_SHM_CAP_GB="$ARENA"; fi
v=$(run_cell "$TAG" "$MODEL" "$SYSTOK" "$SEQ" "$BLIST" "$POL" 2)
echo "SEPPLAN $TAG $MODEL ${SYSTOK} s=$SEQ arena=$ARENA -> $v $(date +%H:%M:%S)" >> "$S2"
echo "$v"
