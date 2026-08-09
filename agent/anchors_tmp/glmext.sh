#!/bin/bash
# glmext.sh — GLM turning-point EXTENSION (2026-08-04 user directive: the GLM
# panels must show the same story as every other model — rungs BEYOND native
# ctx until rc/uns/uns-off all wall and asym is last standing; the original
# GLMTP campaign stopped at ctx (Flash 192k, Air 128k) so no turning points).
# Serial, SOLO (lane-contention lesson), one cell at a time. Systems per rung:
# rc -> un -> uo -> fd(fsdp2, Flash only; Air fsdp2 is load-phase host-dead at
# 16k => seq-independent OOM, no runs) -> asym T1->T2->T3 promote.
# Phases/tags: x1/x2 Flash 1r/2r, y1/y2 Air 1r/2r (e.g. x1rc256, y2t1160).
set -uo pipefail
export TORCHINDUCTOR_COMPILE_THREADS=1
export GPU=0 HOSTFLOOR=500
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
echo "GLMEXT begin $(date +%H:%M)" >> "$S"

# ext_rung PHASE MODEL SEQ RANKS FSDP2 BL_RC BL_UN BL_UO BL_FD BL_ASYM
ext_rung() {
  local ph="$1" model="$2" seq="$3" ranks="$4" fd="$5" brc="$6" bun="$7" buo="$8" bfd="$9" bas="${10}"
  local ASYM_BE=asym_cpuadamwds; [ "$ranks" = "2" ] && ASYM_BE=asym_sdp2_cpuadamwds
  local T3TOK="${ASYM_BE}|recomp-off-full-fg-ker000-ceil0000-ohbm0"
  local sk=$((seq/1000)) v top2
  local P="none|false|false|false|false|false"
  run_cell "${ph}rc${sk}" "$model" "superoffload_mem|recomp"            "$seq" "$brc" "$P" "$ranks" >/dev/null
  run_cell "${ph}un${sk}" "$model" "superoffload_mem|unsloth"           "$seq" "$bun" "$P" "$ranks" >/dev/null
  run_cell "${ph}uo${sk}" "$model" "superoffload_mem|unsloth-off-ohbm0" "$seq" "$buo" "$P" "$ranks" >/dev/null
  [ "$fd" = "1" ] && run_cell "${ph}fd${sk}" "$model" "fsdp2_offload|recomp" "$seq" "$bfd" "$P" "$ranks" >/dev/null
  v=$(run_cell "${ph}t1${sk}" "$model" "${ASYM_BE}|T1" "$seq" "$bas" "$P" "$ranks")
  if [ "$v" != "TRAINED" ]; then
    top2=$(echo $bas | awk '{print $1, $2}')
    v=$(run_cell "${ph}t2${sk}" "$model" "${ASYM_BE}|T2" "$seq" "$top2" "$P" "$ranks")
    [ "$v" != "TRAINED" ] && run_cell "${ph}t3${sk}" "$model" "$T3TOK" "$seq" "$top2" "$P" "$ranks" >/dev/null
  fi
  echo "EXT-RUNG-DONE ${ph} ${model} s=${seq} $(date +%H:%M)" >> "$S"
}

# ---- X1: Flash 1r (GPU0) ----
export GPU=0 HOSTFLOOR=500; export CUDA_VISIBLE_DEVICES=0; unset GPU_POOL DDP_TIMEOUT || true
ext_rung x1 glm4.7-flash 256000 1 1 "2 1" "1"   "2 1" "2 1" "2 1"
ext_rung x1 glm4.7-flash 320000 1 1 "1"   "1"   "1"   "1"   "2 1"
ext_rung x1 glm4.7-flash 384000 1 1 "1"   "1"   "1"   "1"   "1"
ext_rung x1 glm4.7-flash 448000 1 1 "1"   "1"   "1"   "1"   "1"
echo "X1-DONE $(date +%H:%M)" >> "$S"

# ---- X2: Flash 2r (GPUs 0+1) ----
export GPU="0,1" HOSTFLOOR=1200 GPU_POOL="0,1" DDP_TIMEOUT=1500
export CUDA_VISIBLE_DEVICES="0,1"
ext_rung x2 glm4.7-flash 256000 2 1 "2 1" "2 1" "2 1" "2 1" "3 2"
ext_rung x2 glm4.7-flash 320000 2 1 "1"   "1"   "1"   "1"   "2 1"
ext_rung x2 glm4.7-flash 416000 2 1 "1"   "1"   "1"   "1"   "2 1"
ext_rung x2 glm4.7-flash 512000 2 1 "1"   "1"   "1"   "1"   "1"
echo "X2-DONE $(date +%H:%M)" >> "$S"

# ---- Y1: Air 1r (GPU0; arena cap for ~200 GB banks) ----
export GPU=0 HOSTFLOOR=600 ASYM_ARENA_SHM_CAP_GB=240
export CUDA_VISIBLE_DEVICES=0; unset GPU_POOL DDP_TIMEOUT || true
ext_rung y1 glm4.5-air 160000 1 0 "2 1" "2 1" "1" "-" "2 1"
ext_rung y1 glm4.5-air 192000 1 0 "1"   "1"   "1" "-" "2 1"
ext_rung y1 glm4.5-air 256000 1 0 "1"   "1"   "1" "-" "1"
ext_rung y1 glm4.5-air 320000 1 0 "1"   "1"   "1" "-" "1"
echo "Y1-DONE $(date +%H:%M)" >> "$S"

# ---- Y2: Air 2r ----
export GPU="0,1" HOSTFLOOR=1200 GPU_POOL="0,1" DDP_TIMEOUT=1500 ASYM_ARENA_SHM_CAP_GB=240
export CUDA_VISIBLE_DEVICES="0,1"
ext_rung y2 glm4.5-air 160000 2 0 "1"   "2 1" "1" "-" "3 2"
ext_rung y2 glm4.5-air 192000 2 0 "1"   "1"   "1" "-" "2 1"
ext_rung y2 glm4.5-air 256000 2 0 "1"   "1"   "1" "-" "2 1"
ext_rung y2 glm4.5-air 320000 2 0 "1"   "1"   "1" "-" "1"
echo "Y2-DONE GLMEXT-ALL-DONE $(date +%H:%M)" >> "$S"
