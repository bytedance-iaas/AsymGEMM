#!/bin/bash
# Patch chain: cells discovered missing mid-campaign (phi T2 died at 256k ->
# asym bars at 256k/384k must be T3; add 192k last-stand rung for phi).
# Waits for serial_chain v2's ALL-RUNS-COMPLETE marker; never edits live scripts.
set -uo pipefail
export GPU=0 HOSTFLOOR=1300
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
PH=phi3.5-moe
T3TOK="asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0"

for i in $(seq 1 1800); do grep -q "ALL-RUNS-COMPLETE" "$S" 2>/dev/null && break; sleep 30; done
grep -q "ALL-RUNS-COMPLETE" "$S" || { echo "PATCH-ABORT main chain unfinished $(date +%H:%M)" >> "$S"; exit 1; }
echo "PATCH-CHAIN begin $(date +%H:%M)" >> "$S"

run_cell pht3256 $PH "$T3TOK"                       256000 "1"
run_cell pht3384 $PH "$T3TOK"                       384000 "1"
run_cell phu192  $PH "superoffload_mem|unsloth"     192000 "1"
run_cell phr192  $PH "superoffload_mem|recomp"      192000 "1"
run_cell pht1192 $PH "asym_cpuadamwds|T1"           192000 "1"
run_cell pht2192 $PH "asym_cpuadamwds|T2"           192000 "1"
run_cell pho192  $PH "superoffload_mem|unsloth-off" 192000 "1"
echo "PATCH-CHAIN-DONE $(date +%H:%M)" >> "$S"
