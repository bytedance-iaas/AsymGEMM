#!/bin/bash
set -uo pipefail
export GPU=1 HOSTFLOOR=600
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
M=phi3.5-moe
T3TOK="asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0"

run_cell phu128  $M "superoffload_mem|unsloth"     128000 "3 2 1"
r128=$(run_cell phr128 $M "superoffload_mem|recomp" 128000 "1")
run_cell pht1128 $M "asym_cpuadamwds|T1"           128000 "1"
run_cell pht1256 $M "asym_cpuadamwds|T1"           256000 "1"
run_cell phu256  $M "superoffload_mem|unsloth"     256000 "1"
if [ "$r128" = "TRAINED" ]; then run_cell phr256 $M "superoffload_mem|recomp" 256000 "1"; fi
run_cell pht2256 $M "asym_cpuadamwds|T2"           256000 "1"
run_cell pht1384 $M "asym_cpuadamwds|T1"           384000 "1"
run_cell pht2384 $M "asym_cpuadamwds|T2"           384000 "1"
u384=$(run_cell phu384 $M "superoffload_mem|unsloth" 384000 "1")
if [ "$u384" = "TRAINED" ]; then run_cell phu512 $M "superoffload_mem|unsloth" 512000 "1"; fi
run_cell pho256  $M "superoffload_mem|unsloth-off" 256000 "1"
run_cell pho384  $M "superoffload_mem|unsloth-off" 384000 "1"
run_cell pho512  $M "superoffload_mem|unsloth-off" 512000 "1"
run_cell pht2512 $M "asym_cpuadamwds|T2"           512000 "1"
run_cell pht3512 $M "$T3TOK"                       512000 "1"
echo "PHI-CHAIN-DONE $(date +%H:%M)" >> "$S"; touch "$LOGD/phi.done"
