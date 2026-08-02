#!/bin/bash
# Single serial chain for the tp-figure campaign (v2 after the parallel-chain
# host-contention lesson: one run at a time, one GPU, high host floor).
set -uo pipefail
export GPU=0 HOSTFLOOR=1300
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
MX=mixtral-8x22b; PH=phi3.5-moe
T3TOK="asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0"
echo "SERIAL-CHAIN v2 begin $(date +%H:%M)" >> "$S"

# ---- PHI block (phu128 b3 GOOM already banked; walk resumes at b2) ----
run_cell phu128  $PH "superoffload_mem|unsloth"     128000 "2 1"
r128=$(run_cell phr128 $PH "superoffload_mem|recomp" 128000 "1")
run_cell pht1128 $PH "asym_cpuadamwds|T1"           128000 "1"
run_cell pht1256 $PH "asym_cpuadamwds|T1"           256000 "1"
run_cell phu256  $PH "superoffload_mem|unsloth"     256000 "1"
if [ "$r128" = "TRAINED" ]; then run_cell phr256 $PH "superoffload_mem|recomp" 256000 "1"; fi
run_cell pht2256 $PH "asym_cpuadamwds|T2"           256000 "1"
run_cell pht1384 $PH "asym_cpuadamwds|T1"           384000 "1"
run_cell pht2384 $PH "asym_cpuadamwds|T2"           384000 "1"
u384=$(run_cell phu384 $PH "superoffload_mem|unsloth" 384000 "1")
if [ "$u384" = "TRAINED" ]; then run_cell phu512 $PH "superoffload_mem|unsloth" 512000 "1"; fi
run_cell pho256  $PH "superoffload_mem|unsloth-off" 256000 "1"
run_cell pho384  $PH "superoffload_mem|unsloth-off" 384000 "1"
run_cell pho512  $PH "superoffload_mem|unsloth-off" 512000 "1"
run_cell pht2512 $PH "asym_cpuadamwds|T2"           512000 "1"
run_cell pht3512 $PH "$T3TOK"                       512000 "1"
echo "PHI-BLOCK-DONE $(date +%H:%M)" >> "$S"

# ---- MIXTRAL block ----
run_cell mxu064  $MX "superoffload_mem|unsloth"     64000  "2 1"
run_cell mxr064  $MX "superoffload_mem|recomp"      64000  "1"
run_cell mxu128  $MX "superoffload_mem|unsloth"     128000 "1"
m128=$(run_cell mxr128 $MX "superoffload_mem|recomp" 128000 "1")
run_cell mxt1128 $MX "asym_cpuadamwds|T1"           128000 "1"
run_cell mxt1192 $MX "asym_cpuadamwds|T1"           192000 "1"
run_cell mxu192  $MX "superoffload_mem|unsloth"     192000 "1"
run_cell mxt1256 $MX "asym_cpuadamwds|T1"           256000 "1"
run_cell mxu256  $MX "superoffload_mem|unsloth"     256000 "1"
run_cell mxt2256 $MX "asym_cpuadamwds|T2"           256000 "1"
run_cell mxt2320 $MX "asym_cpuadamwds|T2"           320000 "1"
run_cell mxt3320 $MX "$T3TOK"                       320000 "1"
run_cell mxt3384 $MX "$T3TOK"                       384000 "1"
if [ "$m128" = "TRAINED" ]; then run_cell mxr192 $MX "superoffload_mem|recomp" 192000 "1"; fi
run_cell mxo128  $MX "superoffload_mem|unsloth-off" 128000 "1"
run_cell mxo192  $MX "superoffload_mem|unsloth-off" 192000 "1"
echo "MX-BLOCK-DONE $(date +%H:%M)" >> "$S"

# ---- deep phi crowns ----
run_cell pht3640 $PH "$T3TOK"                       640000 "1"
run_cell pht3768 $PH "$T3TOK"                       768000 "1"
echo "SOLO-CHAIN-DONE ALL-RUNS-COMPLETE $(date +%H:%M)" >> "$S"
