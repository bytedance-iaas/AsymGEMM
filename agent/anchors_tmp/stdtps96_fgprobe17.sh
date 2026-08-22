#!/bin/bash
set -uo pipefail
. /workspace/AsymGEMM-SFT-46/third_party/AsymGEMM/agent/anchors_tmp/stdtps96_lib.sh
echo "=== FG-PROBE17 (LF@ebde34d3 + dataset_info restored) BEGIN $(date '+%F %H:%M:%S') ===" >> "$S"
occupiers_alive() { return 0; }   # occupiers intentionally down for the fg forensics
guard() { return 0; }
POL="none|false|false|false|false|false"
v=$(ONE_RANK_GPU=1 run_cell s96q30t2b032lf2 q3-30b-a3b "asym_cpuadamwds|T2B" 32000 "1" "$POL" 1)
echo "FG-PROBE17 canary, LF@ebde34d3 clean -> $v" >> "$S"
echo "=== FG-PROBE17 DONE $(date '+%F %H:%M:%S') ===" >> "$S"
