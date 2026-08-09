#!/bin/bash
# jamba_2r.sh — Jamba2-Mini 2-rank ladder (jamba_integration.md; GLM-2r recipe:
# DP over full-sequence ranks, GPUs 0+1, global tok/s; asym = sdp2 backend,
# tier ladder T1->T2B->T3 (T2 skipped, incident 9)).
set -uo pipefail
export TORCHINDUCTOR_COMPILE_THREADS=1
export GPU="0,1" HOSTFLOOR=1200 GPU_POOL="0,1" DDP_TIMEOUT=1500
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
export CUDA_VISIBLE_DEVICES="0,1"
echo "JAMBA2R begin $(date +%H:%M)" >> "$S"
P="none|false|false|false|false|false"
BE=asym_sdp2_cpuadamwds
T3TOK="${BE}|recomp-off-full-fg-ker000-ceil0000-ohbm0"

tok() { case "$1" in
  rc) echo "superoffload_mem|recomp";;
  un) echo "superoffload_mem|unsloth";;
  uo) echo "superoffload_mem|unsloth-off-ohbm0";;
esac; }

wk() { local sys="$1" sk="$2" bl="$3"
  run_cell "j2${sys}${sk}" jamba2-mini "$(tok $sys)" "${sk}000" "$bl" "$P" 2; }

asym_rung() { local sk="$1" bl="$2" v top1
  top1=$(echo $bl | awk '{print $1}')
  v=$(run_cell "j2t1_${sk}" jamba2-mini "${BE}|T1" "${sk}000" "$bl" "$P" 2)
  [ "$v" = "TRAINED" ] && return 0
  v=$(run_cell "j2t2b_${sk}" jamba2-mini "${BE}|T2B" "${sk}000" "$top1" "$P" 2)
  [ "$v" = "TRAINED" ] && return 0
  v=$(run_cell "j2t3_${sk}" jamba2-mini "$T3TOK" "${sk}000" "$top1" "$P" 2)
  [ "$v" = "TRAINED" ] && return 0
  echo "WALL j2asym s=${sk}k $(date +%H:%M)" >> "$S"; return 1; }

sys_walk() { local sys="$1"; shift
  local spec sk bl v
  for spec in "$@"; do
    sk="${spec%%:*}"; bl="${spec#*:}"; bl="${bl//,/ }"
    v=$(wk "$sys" "$sk" "$bl")
    [ "$v" = "TRAINED" ] || { echo "WALL j2${sys} s=${sk}k ($v) $(date +%H:%M)" >> "$S"; break; }
  done; }

sys_walk rc 32:8,6,4 64:4,3,2 128:2,1 192:1 256:1 320:1
sys_walk un 32:8,6,4 64:4,3,2 128:2,1 192:1 256:1 320:1 384:1
sys_walk uo 32:8,6,4 64:4,3,2 128:2,1 192:1 256:1 320:1 384:1 448:1
for spec in 32:8,6,4 64:4,3,2 128:2,1 192:1 256:1 320:1 384:1 448:1 512:1; do
  sk="${spec%%:*}"; bl="${spec#*:}"; bl="${bl//,/ }"
  asym_rung "$sk" "$bl" || break
done
echo "JAMBA2R-DONE $(date +%H:%M)" >> "$S"
