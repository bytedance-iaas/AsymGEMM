#!/bin/bash
# Patch 2: phi post-window rungs — asym-only candidates at 192k·b2 and 224k,
# baselines' 224k confirmations. Gated on PROBE-CHAIN-DONE.
set -uo pipefail
export GPU=0 HOSTFLOOR=1300
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
PH=phi3.5-moe
T3TOK="asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0"

for i in $(seq 1 1200); do grep -q "PROBE-CHAIN-DONE" "$S" 2>/dev/null && break; sleep 30; done
grep -q "PROBE-CHAIN-DONE" "$S" || { echo "PATCH2-ABORT $(date +%H:%M)" >> "$S"; exit 1; }
echo "PATCH2-CHAIN begin $(date +%H:%M)" >> "$S"

run_cell pht3192 $PH "$T3TOK"                       192000 "2 1"
run_cell pht2192p $PH "asym_cpuadamwds|T2"          192000 "2"
run_cell pht3224 $PH "$T3TOK"                       224000 "2 1"
run_cell pht2224 $PH "asym_cpuadamwds|T2"           224000 "1"
run_cell pho224  $PH "superoffload_mem|unsloth-off" 224000 "1"
run_cell phu224  $PH "superoffload_mem|unsloth"     224000 "1"
echo "PATCH2-CHAIN-DONE ALL-PROBES-COMPLETE $(date +%H:%M)" >> "$S"
