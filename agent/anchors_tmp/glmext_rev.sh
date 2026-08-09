#!/bin/bash
# glmext_rev.sh — GLM turning-point extension, REVERSE runner (c18 back-runner
# per run_glms.md: c14 runs X1→X2→Y1→Y2 forward; this runs Y2→Y1→X2→X1 so the
# two converge). Ladder cells are IDENTICAL to the SFT-39 glmext.sh; tags are
# b-prefixed (by2rc160 vs c14's y2rc160) so artifacts never collide. Every
# cell is appended to the shared ledger (run_glms.md §Log) with parsed
# metrics for TRAINED cells. Rung-skip: if the ledger already shows c14's
# asym cell for the same phase+seq (rung reached its end forward), skip ours.
# Baseline-still-fits-at-cap extension (+64-96k, doc §3) is handled by the
# operator from the final verdicts, not automated here.
set -uo pipefail
export TORCHINDUCTOR_COMPILE_THREADS=1
export GPU=0 HOSTFLOOR=500
. /workspace/AsymGEMM-SFT-46/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
LEDGER=/workspace/AsymGEMM-SFT-46/third_party/AsymGEMM/agent/impls/run_glms.md
echo "GLMEXT-REV begin $(date +%H:%M)" >> "$S"

led() { echo "$1" >> "$LEDGER"; }

# cellw PHASE TAG MODEL SYSTOK SEQ BLIST RANKS -> verdict; ledger line appended
cellw() { local ph="$1" tag="$2" model="$3" systok="$4" seq="$5" blist="$6" ranks="$7"
  local v line b cell dmodel
  v=$(run_cell "$tag" "$model" "$systok" "$seq" "$blist" "none|false|false|false|false|false" "$ranks")
  line=$(grep -a "CELL ${tag} " "$S" | tail -1)
  b=$(echo "$line" | grep -oE "b=[0-9]+" | tail -1 | cut -d= -f2)
  cell="-"
  if [ "$v" = "TRAINED" ]; then
    dmodel=${model//./_}
    cell=$(python3 scripts/lf/parse_fill_cell.py \
      "$B/${tag}-c18_${dmodel}__b${b}_s${seq}_ga1_drop000" "$ranks" "$seq" "$b" 2>/dev/null | tail -1)
    [ -z "$cell" ] && cell="parse-error"
  fi
  led "- [$(date -u '+%m-%d %H:%MZ')] c18 ${ph} ${tag} ${systok} s=${seq} b=${b:-?} -> ${v} | ${cell}"
  echo "$v"
}

# ext_rung PHASE MODEL SEQ RANKS FSDP2 BL_RC BL_UN BL_UO BL_FD BL_ASYM
ext_rung() {
  local ph="$1" model="$2" seq="$3" ranks="$4" fd="$5" brc="$6" bun="$7" buo="$8" bfd="$9" bas="${10}"
  local sk=$((seq/1000))
  # convergence skip: this rung's asym end already logged — by c14's forward
  # tag (y2t1160-style) OR by our own b-tag from a previous chain instance
  # (resume-after-pause support).
  local cph=${ph#b}
  if grep -qE " b?${cph}(t1|t2|t3)${sk} " "$LEDGER"; then
    echo "RUNG-SKIP ${ph} s=${seq} (rung already logged for ${cph}/${ph} t*${sk}) $(date +%H:%M)" >> "$S"
    led "- [$(date -u '+%m-%d %H:%MZ')] c18 ${ph} s=${seq}: SKIP — rung already complete in ledger"
    return 0
  fi
  local ASYM_BE=asym_cpuadamwds; [ "$ranks" = "2" ] && ASYM_BE=asym_sdp2_cpuadamwds
  local T3TOK="${ASYM_BE}|recomp-off-full-fg-ker000-ceil0000-ohbm0"
  local v top2
  local P="none|false|false|false|false|false"
  cellw "$ph" "${ph}rc${sk}" "$model" "superoffload_mem|recomp"            "$seq" "$brc" "$ranks" >/dev/null
  cellw "$ph" "${ph}un${sk}" "$model" "superoffload_mem|unsloth"           "$seq" "$bun" "$ranks" >/dev/null
  cellw "$ph" "${ph}uo${sk}" "$model" "superoffload_mem|unsloth-off-ohbm0" "$seq" "$buo" "$ranks" >/dev/null
  [ "$fd" = "1" ] && cellw "$ph" "${ph}fd${sk}" "$model" "fsdp2_offload|recomp" "$seq" "$bfd" "$ranks" >/dev/null
  v=$(cellw "$ph" "${ph}t1${sk}" "$model" "${ASYM_BE}|T1" "$seq" "$bas" "$ranks")
  if [ "$v" != "TRAINED" ]; then
    top2=$(echo $bas | awk '{print $1, $2}')
    v=$(cellw "$ph" "${ph}t2${sk}" "$model" "${ASYM_BE}|T2" "$seq" "$top2" "$ranks")
    [ "$v" != "TRAINED" ] && cellw "$ph" "${ph}t3${sk}" "$model" "$T3TOK" "$seq" "$top2" "$ranks" >/dev/null
  fi
  echo "EXT-RUNG-DONE ${ph} ${model} s=${seq} $(date +%H:%M)" >> "$S"
}

# Phase order per user 08-04: Air 1r -> Air 2r -> Flash 1r -> Flash 2r
# (c14 forward runs Flash-first, so the two still converge model-wise).

# ---- Y1: Air 1r (GPU0; arena cap for ~200 GB banks) ----
export GPU=0 HOSTFLOOR=600 ASYM_ARENA_SHM_CAP_GB=240
export CUDA_VISIBLE_DEVICES=0; unset GPU_POOL DDP_TIMEOUT || true
ext_rung by1 glm4.5-air 160000 1 0 "2 1" "2 1" "1" "-" "2 1"
ext_rung by1 glm4.5-air 192000 1 0 "1"   "1"   "1" "-" "2 1"
ext_rung by1 glm4.5-air 256000 1 0 "1"   "1"   "1" "-" "1"
ext_rung by1 glm4.5-air 320000 1 0 "1"   "1"   "1" "-" "1"
echo "Y1-DONE(rev) $(date +%H:%M)" >> "$S"
led "- [$(date -u '+%m-%d %H:%MZ')] c18: PHASE-COMPLETE by1 (Air 1r)"

# ---- Y2: Air 2r (GPUs 0+1) ----
export GPU="0,1" HOSTFLOOR=1200 GPU_POOL="0,1" DDP_TIMEOUT=1500 ASYM_ARENA_SHM_CAP_GB=240
export CUDA_VISIBLE_DEVICES="0,1"
ext_rung by2 glm4.5-air 160000 2 0 "1"   "2 1" "1" "-" "3 2"
ext_rung by2 glm4.5-air 192000 2 0 "1"   "1"   "1" "-" "2 1"
ext_rung by2 glm4.5-air 256000 2 0 "1"   "1"   "1" "-" "2 1"
ext_rung by2 glm4.5-air 320000 2 0 "1"   "1"   "1" "-" "1"
echo "Y2-DONE(rev) $(date +%H:%M)" >> "$S"
led "- [$(date -u '+%m-%d %H:%MZ')] c18: PHASE-COMPLETE by2 (Air 2r)"

# ---- X1: Flash 1r (GPU0) ----
export GPU=0 HOSTFLOOR=500; export CUDA_VISIBLE_DEVICES=0
unset GPU_POOL DDP_TIMEOUT ASYM_ARENA_SHM_CAP_GB || true
ext_rung bx1 glm4.7-flash 256000 1 1 "2 1" "1"   "2 1" "2 1" "2 1"
ext_rung bx1 glm4.7-flash 320000 1 1 "1"   "1"   "1"   "1"   "2 1"
ext_rung bx1 glm4.7-flash 384000 1 1 "1"   "1"   "1"   "1"   "1"
ext_rung bx1 glm4.7-flash 448000 1 1 "1"   "1"   "1"   "1"   "1"
echo "X1-DONE(rev) $(date +%H:%M)" >> "$S"
led "- [$(date -u '+%m-%d %H:%MZ')] c18: PHASE-COMPLETE bx1 (Flash 1r)"

# ---- X2: Flash 2r (GPUs 0+1) ----
export GPU="0,1" HOSTFLOOR=1200 GPU_POOL="0,1" DDP_TIMEOUT=1500
export CUDA_VISIBLE_DEVICES="0,1"
ext_rung bx2 glm4.7-flash 256000 2 1 "2 1" "2 1" "2 1" "2 1" "3 2"
ext_rung bx2 glm4.7-flash 320000 2 1 "1"   "1"   "1"   "1"   "2 1"
ext_rung bx2 glm4.7-flash 416000 2 1 "1"   "1"   "1"   "1"   "2 1"
ext_rung bx2 glm4.7-flash 512000 2 1 "1"   "1"   "1"   "1"   "1"
echo "X2-DONE(rev) GLMEXT-REV-ALL-DONE $(date +%H:%M)" >> "$S"
led "- [$(date -u '+%m-%d %H:%MZ')] c18: PHASE-COMPLETE bx2 (Flash 2r) — GLMEXT-REV ALL DONE"
