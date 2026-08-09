#!/bin/bash
# glmext2.sh — GLM turning-point chain, stage 2 (2026-08-05).
# Takes over from glmext.sh at the X2 boundary (glmext killed at ^X2-DONE):
#  X1E  Flash 1r extension — un/uo still alive at 448k → walk them (and asym)
#       up 512k→896k until each walls; + un192 coherence re-probe (old banked
#       un@192k OOM is stale: un now trains 256-448k b1 — non-monotone).
#  X2E  Flash 2r extension — DYNAMIC: any system alive at X2's 512k cap walks
#       up from 576k until it walls.
#  Y1/Y2 Air ladders (from glmext.sh) + survivor walk-up past 320k.
# Rule (user): a model's story is done only when EVERY baseline has a
# measured wall and asym trains beyond it. walk_up stops per-system at its
# first OOM; asym promotes T1→T2→T3 before a wall counts.
set -uo pipefail
export TORCHINDUCTOR_COMPILE_THREADS=1
export GPU=0 HOSTFLOOR=500
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
GLMS_DOC=/workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/impls/run_glms.md
echo "GLMEXT2 begin $(date +%H:%M)" >> "$S"
P="none|false|false|false|false|false"

tok() { case "$1" in
  rc) echo "superoffload_mem|recomp";;
  un) echo "superoffload_mem|unsloth";;
  uo) echo "superoffload_mem|unsloth-off-ohbm0";;
  fd) echo "fsdp2_offload|recomp";;
esac; }

# walk_up PH SYS MODEL RANKS "SEQ1 SEQ2..." [BLIST] — stop at first OOM
walk_up() { local ph="$1" sys="$2" model="$3" ranks="$4" seqs="$5" bl="${6:-1}" s v
  for s in $seqs; do
    v=$(run_cell "${ph}${sys}$((s/1000))" "$model" "$(tok $sys)" "$s" "$bl" "$P" "$ranks")
    [ "$v" = "TRAINED" ] || { echo "WALL ${ph}${sys} ${model} r${ranks} s=${s} ($v) $(date +%H:%M)" >> "$S"; break; }
  done; }

# asym_up PH MODEL RANKS "SEQ1..." [BLIST] — T1→T2→T3 promote per rung
asym_up() { local ph="$1" model="$2" ranks="$3" seqs="$4" bl="${5:-1}" s v top2 be t3
  be=asym_cpuadamwds; [ "$ranks" = "2" ] && be=asym_sdp2_cpuadamwds
  t3="${be}|recomp-off-full-fg-ker000-ceil0000-ohbm0"
  for s in $seqs; do
    v=$(run_cell "${ph}t1$((s/1000))" "$model" "${be}|T1" "$s" "$bl" "$P" "$ranks")
    if [ "$v" != "TRAINED" ]; then
      top2=$(echo $bl | awk '{print $1, $2}')
      v=$(run_cell "${ph}t2$((s/1000))" "$model" "${be}|T2" "$s" "$top2" "$P" "$ranks")
      [ "$v" != "TRAINED" ] && v=$(run_cell "${ph}t3$((s/1000))" "$model" "$t3" "$s" "$top2" "$P" "$ranks")
    fi
    [ "$v" = "TRAINED" ] || { echo "WALL ${ph}asym ${model} r${ranks} s=${s} ($v) $(date +%H:%M)" >> "$S"; break; }
  done; }

# alive PH SYS SEQK — was the system's LAST verdict at that rung TRAINED?
alive() { grep -a "CELL ${1}${2}${3} " "$S" | tail -1 | grep -q 'TRAINED'; }

claimed_other() { grep -aE "CLAIM.*$1" "$GLMS_DOC" 2>/dev/null | grep -qv 'c14 session'; }

# ---- X1E: Flash 1r ----
export GPU=0 HOSTFLOOR=500 CUDA_VISIBLE_DEVICES=0; unset GPU_POOL DDP_TIMEOUT || true
run_cell x1eun192 glm4.7-flash "$(tok un)" 192000 "1" "$P" 1 >/dev/null   # coherence re-probe
walk_up x1e un glm4.7-flash 1 "512000 576000 640000 704000 768000 896000"
walk_up x1e uo glm4.7-flash 1 "512000 576000 640000 704000 768000 896000"
asym_up x1e glm4.7-flash 1 "512000 576000 640000 704000 768000 896000"
echo "X1E-DONE $(date +%H:%M)" >> "$S"

# ---- X2: Flash 2r BASE ladder (moved here from glmext.sh — that chain was
# killed mid-x2rc256-b1 to prioritize the 1r story; rc256 b2 GOOM already
# measured, so rc's 256k blist starts at b1) ----
export GPU="0,1" HOSTFLOOR=1200 GPU_POOL="0,1" DDP_TIMEOUT=1500 CUDA_VISIBLE_DEVICES="0,1"
walk_up x2 rc glm4.7-flash 2 "256000"
walk_up x2 rc glm4.7-flash 2 "320000 416000 512000"
walk_up x2 un glm4.7-flash 2 "256000" "2 1"
walk_up x2 un glm4.7-flash 2 "320000 416000 512000"
walk_up x2 uo glm4.7-flash 2 "256000" "2 1"
walk_up x2 uo glm4.7-flash 2 "320000 416000 512000"
walk_up x2 fd glm4.7-flash 2 "256000" "2 1"
walk_up x2 fd glm4.7-flash 2 "320000 416000 512000"
asym_up x2 glm4.7-flash 2 "256000" "3 2"
asym_up x2 glm4.7-flash 2 "320000 416000" "2 1"
asym_up x2 glm4.7-flash 2 "512000"
echo "X2-DONE $(date +%H:%M)" >> "$S"

# ---- X2E: Flash 2r dynamic extension (survivors of the 512k cap) ----
X2EXT="576000 640000 704000 832000"
for sys in rc un uo fd; do
  alive x2 $sys 512 && walk_up x2e $sys glm4.7-flash 2 "$X2EXT"
done
if alive x2 t1 512 || alive x2 t2 512 || alive x2 t3 512; then
  asym_up x2e glm4.7-flash 2 "$X2EXT"
fi
echo "X2E-DONE FLASH-ALL-DONE $(date +%H:%M)" >> "$S"

# ---- Y1: Air 1r (skip if another runner claimed it) ----
if claimed_other Y1; then echo "Y1-SKIP claimed elsewhere $(date +%H:%M)" >> "$S"; else
export GPU=0 HOSTFLOOR=600 ASYM_ARENA_SHM_CAP_GB=240 CUDA_VISIBLE_DEVICES=0
unset GPU_POOL DDP_TIMEOUT || true
for s in 160000 192000; do
  walk_up y1 rc glm4.5-air 1 "$s" "2 1"; done
walk_up y1 rc glm4.5-air 1 "256000 320000"
walk_up y1 un glm4.5-air 1 "160000 192000" "2 1"
walk_up y1 un glm4.5-air 1 "256000 320000"
walk_up y1 uo glm4.5-air 1 "160000 192000 256000 320000"
asym_up y1 glm4.5-air 1 "160000 192000" "2 1"
asym_up y1 glm4.5-air 1 "256000 320000 384000 448000"
echo "Y1-DONE $(date +%H:%M)" >> "$S"; fi

# ---- Y2: Air 2r ----
if claimed_other Y2; then echo "Y2-SKIP claimed elsewhere $(date +%H:%M)" >> "$S"; else
export GPU="0,1" HOSTFLOOR=1200 GPU_POOL="0,1" DDP_TIMEOUT=1500 ASYM_ARENA_SHM_CAP_GB=240
export CUDA_VISIBLE_DEVICES="0,1"
walk_up y2 rc glm4.5-air 2 "160000 192000 256000 320000"
walk_up y2 un glm4.5-air 2 "160000" "2 1"
walk_up y2 un glm4.5-air 2 "192000 256000 320000"
walk_up y2 uo glm4.5-air 2 "160000 192000 256000 320000"
asym_up y2 glm4.5-air 2 "160000 192000" "3 2"
asym_up y2 glm4.5-air 2 "256000 320000 384000 448000" "2 1"
echo "Y2-DONE $(date +%H:%M)" >> "$S"; fi
echo "GLMEXT2-ALL-DONE $(date +%H:%M)" >> "$S"
