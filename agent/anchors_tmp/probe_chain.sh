#!/bin/bash
# Final probe round for the two figure panels: best-batch fills + deep walls.
set -uo pipefail
export GPU=0 HOSTFLOOR=1300
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
MX=mixtral-8x22b; PH=phi3.5-moe
T3TOK="asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0"
echo "PROBE-CHAIN begin $(date +%H:%M)" >> "$S"

# phi 128k asym best-batch (T1 47.4 GiB @ b1 -> b3 should fit)
run_cell pht1128 $PH "asym_cpuadamwds|T1" 128000 "3 2"

# mixtral deep walls
run_cell mxu320  $MX "superoffload_mem|unsloth" 320000 "1"
run_cell mxt1320 $MX "asym_cpuadamwds|T1"       320000 "1"
run_cell mxt2352 $MX "asym_cpuadamwds|T2"       352000 "1"
run_cell mxt3352 $MX "$T3TOK"                   352000 "1"

# mixtral 64k best-batch fills
v=$(run_cell mxu064p $MX "superoffload_mem|unsloth" 64000 "3")
if [ "$v" = "TRAINED" ]; then run_cell mxu064p $MX "superoffload_mem|unsloth" 64000 "4"; fi
run_cell mxt1064 $MX "asym_cpuadamwds|T1" 64000 "3"
run_cell mxt2064 $MX "asym_cpuadamwds|T2" 64000 "4"
run_cell mxt3064 $MX "$T3TOK"             64000 "3"
run_cell mxr064p $MX "superoffload_mem|recomp" 64000 "2"

# mixtral 128k uns_off best-batch
run_cell mxo128p $MX "superoffload_mem|unsloth-off" 128000 "2"
echo "PROBE-CHAIN-DONE $(date +%H:%M)" >> "$S"
