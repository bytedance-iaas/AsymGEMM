#!/bin/bash
# jamba_1r.sh — Jamba2-Mini 1-rank throughput ladder (jamba_integration.md).
# House recipe: per-system walk-up with batch walk-lists (best-over-batch),
# asym = 4-tier promotion T1->T2->T2B->T3 per rung; walls measured; serial
# SOLO GPU0. Rungs to 448k (beyond ctx 262k legal: 4 rope attn layers, mamba
# length-agnostic, no windowing); extend on survivors.
set -uo pipefail
export TORCHINDUCTOR_COMPILE_THREADS=1
export GPU=0 HOSTFLOOR=500
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
echo "JAMBA1R begin $(date +%H:%M)" >> "$S"
P="none|false|false|false|false|false"
export CUDA_VISIBLE_DEVICES=0
BE=asym_cpuadamwds
T3TOK="${BE}|recomp-off-full-fg-ker000-ceil0000-ohbm0"

tok() { case "$1" in
  rc) echo "superoffload_mem|recomp";;
  un) echo "superoffload_mem|unsloth";;
  uo) echo "superoffload_mem|unsloth-off-ohbm0";;
esac; }

# wk SYS SEQK BLIST -> run cell, return verdict
wk() { local sys="$1" sk="$2" bl="$3"
  run_cell "j1${sys}${sk}" jamba2-mini "$(tok $sys)" "${sk}000" "$bl" "$P" 1; }

# asym rung with 4-tier promotion; blist applies to T1, top-1 to deeper tiers
asym_rung() { local sk="$1" bl="$2" v top1
  top1=$(echo $bl | awk '{print $1}')
  v=$(run_cell "j1t1_${sk}" jamba2-mini "${BE}|T1" "${sk}000" "$bl" "$P" 1)
  [ "$v" = "TRAINED" ] && return 0
  v=$(run_cell "j1t2_${sk}" jamba2-mini "${BE}|T2" "${sk}000" "$top1" "$P" 1)
  [ "$v" = "TRAINED" ] && return 0
  v=$(run_cell "j1t2b_${sk}" jamba2-mini "${BE}|T2B" "${sk}000" "$top1" "$P" 1)
  [ "$v" = "TRAINED" ] && return 0
  v=$(run_cell "j1t3_${sk}" jamba2-mini "$T3TOK" "${sk}000" "$top1" "$P" 1)
  [ "$v" = "TRAINED" ] && return 0
  echo "WALL j1asym s=${sk}k $(date +%H:%M)" >> "$S"; return 1; }

# ---- per-system walk-ups (stop at first full-list OOM rung) ----
sys_walk() { local sys="$1"; shift
  local spec sk bl v
  for spec in "$@"; do
    sk="${spec%%:*}"; bl="${spec#*:}"; bl="${bl//,/ }"
    v=$(wk "$sys" "$sk" "$bl")
    [ "$v" = "TRAINED" ] || { echo "WALL j1${sys} s=${sk}k ($v) $(date +%H:%M)" >> "$S"; break; }
  done; }

sys_walk rc 32:8,6,4 64:4,3,2 128:2,1 192:1 256:1 320:1
sys_walk un 32:8,6,4 64:4,3,2 128:2,1 192:1 256:1 320:1 384:1
sys_walk uo 32:8,6,4 64:4,3,2 128:2,1 192:1 256:1 320:1 384:1 448:1

# ---- asym ladder (batch walks at short rungs; tiers carry the deep end) ----
for spec in 32:8,6,4 64:4,3,2 128:2,1 192:1 256:1 320:1 384:1 448:1; do
  sk="${spec%%:*}"; bl="${spec#*:}"; bl="${bl//,/ }"
  asym_rung "$sk" "$bl" || break
done
echo "JAMBA1R-DONE $(date +%H:%M)" >> "$S"
