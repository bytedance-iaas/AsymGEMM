#!/bin/bash
# stdtps46_a1_q122_2r.sh — Agent-1 GAP SWEEP by Session D (c18 = the node the
# banked 122B cells were measured on): the Qwen3.5-122B-A10B 2-rank main-grid
# gaps 160k/224k (standardize_tps.md grid 160..320k @32k). Claimed in the
# doc's LIVE CLAIMS 12:4x (Session A's staged s2*122 lines to be dropped).
# Protocol = the banked fv5/fv6/fv7/fv8/fv9 + fz2r12k cells: GPUs 0+1, w1+m2,
# GLOBAL tok/s, default watchdog floor (50), baselines superoffload_mem
# recomp / unsloth-ohbm0, zero3_offload_mem|recomp, asym = sEP
# asym_sepplan2_cpuadamwds|T1 with ASYM_ARENA_SHM_CAP_GB=345 (grad-offload
# off — the 320k/336k grad-offload stack is a deeper-rung lever, not used at
# 128k-288k). Batch walks seeded from the nearest banked rung (128k = b2).
set -uo pipefail
. /workspace/AsymGEMM-SFT-46/third_party/AsymGEMM/agent/anchors_tmp/stdtps46_lib.sh
POL="none|false|false|false|false|false"
M=q3.5-122b-a10b
RC="superoffload_mem|recomp"
UN="superoffload_mem|unsloth-ohbm0"
Z3="zero3_offload_mem|recomp"
SEP="asym_sepplan2_cpuadamwds|T1"
ARENA="ASYM_ARENA_SHM_CAP_GB=345"
echo "=== STDTPS46-A1-Q122-2R BEGIN $(date '+%F %H:%M:%S') ===" >> "$S"
# 160k column (all survivors fit by the banked 192k cells; b2 seeded from 128k)
v=$(run_cell s1q122un160 $M "$UN" 160000 "2 1" "$POL" 2);                 echo "A1-Q122 un@160k -> $v" >> "$S"
v=$(ARM_ENV="$ARENA" run_cell s1q122sep160 $M "$SEP" 160000 "2 1" "$POL" 2); echo "A1-Q122 sEP-T1@160k -> $v" >> "$S"
v=$(run_cell s1q122rc160 $M "$RC" 160000 "1" "$POL" 2);                   echo "A1-Q122 rc@160k -> $v" >> "$S"
v=$(run_cell s1q122z3160 $M "$Z3" 160000 "1" "$POL" 2);                   echo "A1-Q122 zero3@160k -> $v" >> "$S"
# 224k column (un/asym fit by the banked 256k cells; rc/zero3 inside their (192k,256k] bracket -> probes)
v=$(run_cell s1q122un224 $M "$UN" 224000 "1" "$POL" 2);                   echo "A1-Q122 un@224k -> $v" >> "$S"
v=$(ARM_ENV="$ARENA" run_cell s1q122sep224 $M "$SEP" 224000 "1" "$POL" 2); echo "A1-Q122 sEP-T1@224k -> $v" >> "$S"
v=$(run_cell s1q122rc224 $M "$RC" 224000 "1" "$POL" 2);                   echo "A1-Q122 rc@224k PROBE -> $v" >> "$S"
v=$(run_cell s1q122z3224 $M "$Z3" 224000 "1" "$POL" 2);                   echo "A1-Q122 zero3@224k PROBE -> $v" >> "$S"
echo "=== STDTPS46-A1-Q122-2R DONE $(date '+%F %H:%M:%S') ===" >> "$S"
