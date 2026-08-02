#!/bin/bash
# MERGE VALIDATION QUEUE 2 (merge_progress.md Phase 3-ext, 2026-08-02):
# 4 models x 2 near-capacity cells, serial GPU0, fresh tags, one at a time.
# refs: X1/X2 llama 9-row matrix rows 5/6 (548/545 · 171.1 · ~963-982; 280/279 · 182.4 · ~976-983)
#       X3 flash T3 192k·b5 (158.8 · 723 · 1.322/1.226/1.235, c14) X4 flash T1 192k·b2 (f1s_t1192 artifacts, eff ~812)
#       X5 mixtral T2 320k·b1 (mxt2320 artifacts, eff ~670, RSS ~880-910) X6 mixtral T3 64k·b2 (anchor at3b2: 58.7 · 2534 compute · 908)
#       X7/X8 q3.5-122b T2 448k·b1 (520.2 s/it · 861 · 171.3 · 846, c18) / 480k·b1 (563.6 · 852 · 177.8 96%-edge · 849, c18)
set -uo pipefail
export GPU=0 HOSTFLOOR=1300
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
T3TOK="asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0"
echo "MRG-VAL2 begin $(date +%H:%M)" >> "$S"

# X1/X2 floor=25: llama refs (RSS 963-983) come from c06's roomier host pool; on c14 the usable
# CPU pool is the 960-GB Grace zones (idle HBM NUMA free is NOT MemAvailable) and the cell's
# NORMAL operating point is ~50-60 GB avail — the default floor=50 clips it mid-creep.
# Trainer metrics at x1llt2/x1b were reference-grade (171.1 GiB byte-exact, RSS 893 leaner).
HOST_MEM_WATCHDOG_FLOOR_GB=25 MAX_SAMPLES=1024 run_cell x1d llama3.3-70b "asym_cpuadamwds|T2" 192000 "2"
echo "MRG-X1-DONE $(date +%H:%M)" >> "$S"
HOST_MEM_WATCHDOG_FLOOR_GB=25 run_cell x2d llama3.3-70b "asym_cpuadamwds|T2" 448000 "1"
echo "MRG-X2-DONE $(date +%H:%M)" >> "$S"
run_cell x3f47t3 glm4.7-flash "$T3TOK" 192000 "5"
echo "MRG-X3-DONE $(date +%H:%M)" >> "$S"
run_cell x4f47t1 glm4.7-flash "asym_cpuadamwds|T1" 192000 "2"
echo "MRG-X4-DONE $(date +%H:%M)" >> "$S"
run_cell x5mxt2 mixtral-8x22b "asym_cpuadamwds|T2" 320000 "1"
echo "MRG-X5-DONE $(date +%H:%M)" >> "$S"
run_cell x6mxt3 mixtral-8x22b "$T3TOK" 64000 "2"
echo "MRG-X6-DONE $(date +%H:%M)" >> "$S"
run_cell x7q122 q3.5-122b-a10b "asym_cpuadamwds|T2" 448000 "1"
echo "MRG-X7-DONE $(date +%H:%M)" >> "$S"
run_cell x8q122 q3.5-122b-a10b "asym_cpuadamwds|T2" 480000 "1"
echo "MRG-X8-DONE $(date +%H:%M)" >> "$S"
echo "MRG-VAL2-DONE ALL-8-COMPLETE $(date +%H:%M)" >> "$S"
