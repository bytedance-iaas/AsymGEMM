#!/bin/bash
# stdtps46_a1_q30_2r.sh — Agent-1 GAP SWEEP by Session D (c18): Qwen3-30B-A3B
# 2-rank main-grid cells (standardize_tps.md grid 384..1024k @128k; banked 2r
# seqs 384k/640k/720k/800k/880k/960k/1.04M/1.12M). Claimed in the doc's LIVE
# CLAIMS banner 15:5x. Protocol = the banked R2A/ext cells (sep2t2-c14,
# f2sep384g30, dp2uns-c14, ff2a384c): GPUs 0+1, w1+m2, GLOBAL tok/s, default
# floor, baselines superoffload_mem unsloth-ohbm0 / fsdp2_offload|recomp,
# asym = sEP asym_sepplan2_cpuadamwds|T2 (keep-acts trio, arena 160 default).
# Required: un@512k (fits by 640k), sEP-T2@512k, fsdp2@512k PROBE (384k fit /
# 640k OOM bracket), sEP-T2@768k/896k/1.02M (banked neighbours 720k-1.04M).
# rc/zero3 beyond (384k,392k] and uo beyond (384k,640k]-bracket (probe, lean).
# Upgrade: sEP-T2@384k b2 up-probe (banked b1 2314 sits at 50% HBM).
set -uo pipefail
. /workspace/AsymGEMM-SFT-46/third_party/AsymGEMM/agent/anchors_tmp/stdtps46_lib.sh
POL="none|false|false|false|false|false"
M=q3-30b-a3b
UN="superoffload_mem|unsloth-ohbm0"
UO="superoffload_mem|unsloth-off-ohbm0"
FD="fsdp2_offload|recomp"
SEP="asym_sepplan2_cpuadamwds|T2"
echo "=== STDTPS46-A1-Q30-2R BEGIN $(date '+%F %H:%M:%S') ===" >> "$S"
v=$(run_cell s1q30un512 $M "$UN" 512000 "1" "$POL" 2);      echo "A1-Q30 un@512k -> $v" >> "$S"
v=$(run_cell s1q30sep512 $M "$SEP" 512000 "1" "$POL" 2);    echo "A1-Q30 sEP-T2@512k -> $v" >> "$S"
v=$(run_cell s1q30fd512 $M "$FD" 512000 "1" "$POL" 2);      echo "A1-Q30 fsdp2@512k PROBE -> $v" >> "$S"
v=$(run_cell s1q30sep768 $M "$SEP" 768000 "1" "$POL" 2);    echo "A1-Q30 sEP-T2@768k -> $v" >> "$S"
v=$(run_cell s1q30sep896 $M "$SEP" 896000 "1" "$POL" 2);    echo "A1-Q30 sEP-T2@896k -> $v" >> "$S"
v=$(run_cell s1q30sep1024 $M "$SEP" 1024000 "1" "$POL" 2);  echo "A1-Q30 sEP-T2@1.02M -> $v" >> "$S"
# upgrade + lean probe
v=$(run_cell s1q30sep384 $M "$SEP" 384000 "2" "$POL" 2);    echo "A1-Q30 sEP-T2@384k b2 up-probe -> $v" >> "$S"
v=$(run_cell s1q30uo512 $M "$UO" 512000 "1" "$POL" 2);      echo "A1-Q30 uo@512k PROBE (lean) -> $v" >> "$S"
echo "=== STDTPS46-A1-Q30-2R DONE $(date '+%F %H:%M:%S') ===" >> "$S"
