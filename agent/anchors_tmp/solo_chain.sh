#!/bin/bash
set -uo pipefail
export GPU=0 HOSTFLOOR=900
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
for i in $(seq 1 1500); do [ -f "$LOGD/mx.done" ] && [ -f "$LOGD/phi.done" ] && break; sleep 30; done
{ [ -f "$LOGD/mx.done" ] && [ -f "$LOGD/phi.done" ]; } || { echo "SOLO-ABORT chains unfinished $(date +%H:%M)" >> "$S"; exit 1; }
n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
[ "$n" -eq 0 ] && rm -f /dev/shm/asym_fabric_* 2>/dev/null

run_cell mxo128  mixtral-8x22b "superoffload_mem|unsloth-off" 128000 "1"
run_cell mxo192  mixtral-8x22b "superoffload_mem|unsloth-off" 192000 "1"
T3TOK="asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0"
run_cell pht3640 phi3.5-moe "$T3TOK" 640000 "1"
run_cell pht3768 phi3.5-moe "$T3TOK" 768000 "1"
echo "SOLO-CHAIN-DONE ALL-RUNS-COMPLETE $(date +%H:%M)" >> "$S"; touch "$LOGD/solo.done"
