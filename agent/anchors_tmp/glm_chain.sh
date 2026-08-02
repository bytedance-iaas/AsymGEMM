#!/bin/bash
# GLM validation campaign (model_integration.md protocol, user go 2026-07-28):
# dev loss-parity pairs (8k·b1 w1+m1) then memory validation pairs
# (uns-off-lean vs asym T3, w1+m2) — Flash 192k·b2(+b3), Air 128k·b2(+b3).
# Serial, GPU0, all ops lessons applied (fresh tags, OOM-first verdicts).
set -uo pipefail
export GPU=0 HOSTFLOOR=1300
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
AIR=glm4.5-air; FLASH=glm4.7-flash
T3TOK="asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0"
UNSOFF="superoffload_mem|unsloth-off-ohbm0"
echo "GLM-CHAIN begin $(date +%H:%M)" >> "$S"

# --- dev pairs (w1+m1) ---
MAX_STEPS=1 run_cell gdev47u $FLASH "superoffload_mem|unsloth" 8000 "1"
MAX_STEPS=1 run_cell gdev47a $FLASH "asym_cpuadamwds|T1"       8000 "1"
MAX_STEPS=1 run_cell gdev45u $AIR   "superoffload_mem|unsloth" 8000 "1"
MAX_STEPS=1 run_cell gdev45a $AIR   "asym_cpuadamwds|T1"       8000 "1"
echo "GLM-DEV-DONE $(date +%H:%M)" >> "$S"

# --- Flash validation: 192k·b2 pair, walker down on OOM, b3 capacity probes ---
o2=$(run_cell gval47o $FLASH "$UNSOFF" 192000 "2 1")
t2=$(run_cell gval47t $FLASH "$T3TOK"  192000 "2 1")
if [ "$o2" = "TRAINED" ]; then run_cell gval47o3 $FLASH "$UNSOFF" 192000 "3"; fi
if [ "$t2" = "TRAINED" ]; then run_cell gval47t3 $FLASH "$T3TOK"  192000 "3"; fi
echo "GLM-FLASH-VAL-DONE $(date +%H:%M)" >> "$S"

# --- Air validation: 128k·b2 pair, walker down on OOM, b3 capacity probes ---
oa=$(run_cell gval45o $AIR "$UNSOFF" 128000 "2 1")
ta=$(run_cell gval45t $AIR "$T3TOK"  128000 "2 1")
if [ "$oa" = "TRAINED" ]; then run_cell gval45o3 $AIR "$UNSOFF" 128000 "3"; fi
if [ "$ta" = "TRAINED" ]; then run_cell gval45t3 $AIR "$T3TOK"  128000 "3"; fi
echo "GLM-CHAIN-DONE ALL-GLM-RUNS-COMPLETE $(date +%H:%M)" >> "$S"
