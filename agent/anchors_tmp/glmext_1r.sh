#!/bin/bash
# glmext_1r.sh — 1-RANK ONLY (2026-08-05 user: fix one panel at a time; no 2r
# or other jobs until the 1r plot is done). Flash 1r walk-to-wall (un/uo/asym
# past the 448k cap; rc walled >320k, fd walled >384k — measured) + un192
# coherence re-probe, THEN Air 1r ladder. Markers: X1E-DONE, Y1-DONE,
# GLM1R-ALL-DONE. Everything serial+solo, GPU0 only.
set -uo pipefail
export TORCHINDUCTOR_COMPILE_THREADS=1
export GPU=0 HOSTFLOOR=500
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
echo "GLM1R begin $(date +%H:%M)" >> "$S"
P="none|false|false|false|false|false"

tok() { case "$1" in
  rc) echo "superoffload_mem|recomp";;
  un) echo "superoffload_mem|unsloth";;
  uo) echo "superoffload_mem|unsloth-off-ohbm0";;
  fd) echo "fsdp2_offload|recomp";;
esac; }

walk_up() { local ph="$1" sys="$2" model="$3" ranks="$4" seqs="$5" bl="${6:-1}" s v
  for s in $seqs; do
    v=$(run_cell "${ph}${sys}$((s/1000))" "$model" "$(tok $sys)" "$s" "$bl" "$P" "$ranks")
    [ "$v" = "TRAINED" ] || { echo "WALL ${ph}${sys} ${model} r${ranks} s=${s} ($v) $(date +%H:%M)" >> "$S"; break; }
  done; }

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

# ---- X1E: Flash 1r to the walls ----
export CUDA_VISIBLE_DEVICES=0
run_cell x1eun192 glm4.7-flash "$(tok un)" 192000 "1" "$P" 1 >/dev/null  # stale-OOM re-probe
walk_up x1e un glm4.7-flash 1 "512000 576000 640000 704000 768000 896000"
walk_up x1e uo glm4.7-flash 1 "512000 576000 640000 704000 768000 896000"
asym_up x1e glm4.7-flash 1 "512000 576000 640000 704000 768000 896000"
echo "X1E-DONE $(date +%H:%M)" >> "$S"

# ---- Y1: Air 1r ladder (walls + asym walk) ----
export HOSTFLOOR=600 ASYM_ARENA_SHM_CAP_GB=240
walk_up y1 rc glm4.5-air 1 "160000 192000" "2 1"
walk_up y1 rc glm4.5-air 1 "256000 320000"
walk_up y1 un glm4.5-air 1 "160000 192000" "2 1"
walk_up y1 un glm4.5-air 1 "256000 320000"
walk_up y1 uo glm4.5-air 1 "160000 192000 256000 320000"
asym_up y1 glm4.5-air 1 "160000 192000" "2 1"
asym_up y1 glm4.5-air 1 "256000 320000 384000 448000"
echo "Y1-DONE GLM1R-ALL-DONE $(date +%H:%M)" >> "$S"
