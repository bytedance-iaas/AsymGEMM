#!/bin/bash
set -uo pipefail
export GPU=0 HOSTFLOOR=1000
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
M=mixtral-8x22b
T3TOK="asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0"

run_cell mxu064  $M "superoffload_mem|unsloth"     64000  "2 1"
run_cell mxr064  $M "superoffload_mem|recomp"      64000  "1"
run_cell mxu128  $M "superoffload_mem|unsloth"     128000 "1"
r128=$(run_cell mxr128 $M "superoffload_mem|recomp" 128000 "1")
run_cell mxt1128 $M "asym_cpuadamwds|T1"           128000 "1"
run_cell mxt1192 $M "asym_cpuadamwds|T1"           192000 "1"
run_cell mxu192  $M "superoffload_mem|unsloth"     192000 "1"
run_cell mxt1256 $M "asym_cpuadamwds|T1"           256000 "1"
run_cell mxu256  $M "superoffload_mem|unsloth"     256000 "1"
run_cell mxt2256 $M "asym_cpuadamwds|T2"           256000 "1"
run_cell mxt2320 $M "asym_cpuadamwds|T2"           320000 "1"
run_cell mxt3320 $M "$T3TOK"                       320000 "1"
run_cell mxt3384 $M "$T3TOK"                       384000 "1"
if [ "$r128" = "TRAINED" ]; then run_cell mxr192 $M "superoffload_mem|recomp" 192000 "1"; fi
echo "MX-CHAIN-DONE $(date +%H:%M)" >> "$S"; touch "$LOGD/mx.done"
