#!/bin/bash
# F1 SOLO REDO: re-measure contended best-batch cells one-at-a-time (lesson:
# parallel lanes corrupt throughput; fits/GOOMs stay valid). Gated on F2-DONE.
# Solo-clean cells kept from F1: all of 32k + 96k rungs; 192k uo/T3 (gw-era).
set -uo pipefail
export GPU=0 HOSTFLOOR=1300 TORCHINDUCTOR_COMPILE_THREADS=1
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
T3TOK="asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0"
for i in $(seq 1 2880); do grep -q "F2-DONE" "$S" 2>/dev/null && break; sleep 30; done
grep -q "F2-DONE" "$S" || { echo "F1REDO-ABORT $(date +%H:%M)" >> "$S"; exit 1; }
echo "F1-REDO begin $(date +%H:%M)" >> "$S"
M=glm4.7-flash
run_cell f1s_rc64  $M "superoffload_mem|recomp"            64000  "3"
run_cell f1s_un64  $M "superoffload_mem|unsloth"           64000  "4"
run_cell f1s_uo64  $M "superoffload_mem|unsloth-off-ohbm0" 64000  "6"
run_cell f1s_t164  $M "asym_cpuadamwds|T1"                 64000  "6"
run_cell f1s_rc128 $M "superoffload_mem|recomp"            128000 "2"
run_cell f1s_un128 $M "superoffload_mem|unsloth"           128000 "1"
run_cell f1s_uo128 $M "superoffload_mem|unsloth-off-ohbm0" 128000 "3 2"
run_cell f1s_t1128 $M "asym_cpuadamwds|T1"                 128000 "3"
run_cell f1s_rc160 $M "superoffload_mem|recomp"            160000 "1"
run_cell f1s_un160 $M "superoffload_mem|unsloth"           160000 "1"
run_cell f1s_uo160 $M "superoffload_mem|unsloth-off-ohbm0" 160000 "2"
run_cell f1s_t1160 $M "asym_cpuadamwds|T1"                 160000 "2"
run_cell f1s_rc192 $M "superoffload_mem|recomp"            192000 "1"
run_cell f1s_t1192 $M "asym_cpuadamwds|T1"                 192000 "2"
echo "F1-REDO-DONE $(date +%H:%M)" >> "$S"
