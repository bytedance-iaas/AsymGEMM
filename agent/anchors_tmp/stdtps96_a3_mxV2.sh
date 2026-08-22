#!/bin/bash
# stdtps96_a3_mxV2.sh — METHOD V2 tail of Agent 3: mixtral asym-1r rungs
# 64-128K (the GOOM-prone block, now wedge-safe) + the 2r 64K b1-upgrade
# trio (un/uo/asym). NO occupiers (full 185G card, peak-audit verdicts).
set -uo pipefail
export TORCHINDUCTOR_COMPILE_THREADS=1
export GPU="0" HOSTFLOOR=300 OCC_PIDS=""
. /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/stdtps96_lib2.c14.sh
export GPU_POOL="0"
echo "== STDTPS96-A3-MX-V2 begin (peak-audit mode) $(date '+%m-%d %H:%M')" >> "$S"
MX=mixtral-8x22b; P="none|false|false|false|false|false"; AS1="asym_cpuadamwds|T1"
die() { echo "ABORT-MXV2: $1 $(date '+%m-%d %H:%M')" >> "$S"; exit 1; }
chk() { case "$1" in TRAINED_FIT96*|TRAINED_EDGE96*|OVER96*|GOOM|COOM) :;; *) die "infra verdict $1";; esac; }

v=$(run_cell_pa v2mx1t1_64  $MX "$AS1" 64000  "3 2 1" "$P" 1); chk "$v"
v=$(run_cell_pa v2mx1t1_80  $MX "$AS1" 80000  "2 1"   "$P" 1); chk "$v"
v=$(run_cell_pa v2mx1t1_96  $MX "$AS1" 96000  "2 1"   "$P" 1); chk "$v"
v=$(run_cell_pa v2mx1t1_112 $MX "$AS1" 112000 "2 1"   "$P" 1); chk "$v"
v=$(run_cell_pa v2mx1t1_128 $MX "$AS1" 128000 "2 1"   "$P" 1); chk "$v"

# ---- 2r 64K b1 upgrades (edge-taxed b2 first-fits from occupier mode) ----
export GPU="0,1" GPU_POOL="0,1" CUDA_VISIBLE_DEVICES="0,1" DDP_TIMEOUT=1500 ASYM_ARENA_SHM_CAP_GB=285
v=$(run_cell_pa v2mx2un64b1 $MX "superoffload_mem|unsloth-ohbm0"     64000 "1" "$P" 2); chk "$v"
v=$(run_cell_pa v2mx2uo64b1 $MX "superoffload_mem|unsloth-off-ohbm0" 64000 "1" "$P" 2); chk "$v"
v=$(run_cell_pa v2mx2t164b1 $MX "asym_sdp2_cpuadamwds|T1"            64000 "1" "$P" 2); chk "$v"
echo "== STDTPS96-A3-MX-V2-DONE $(date '+%m-%d %H:%M')" >> "$S"
