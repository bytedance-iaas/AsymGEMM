#!/bin/bash
set -uo pipefail
. /workspace/AsymGEMM-SFT-46/third_party/AsymGEMM/agent/anchors_tmp/stdtps96_lib.sh
echo "=== FG-PROBE16 (-46 + LF pinned ebde34d3) BEGIN $(date '+%F %H:%M:%S') ===" >> "$S"
occupiers_alive() { return 0; }
guard() { return 0; }
POL="none|false|false|false|false|false"
v=$(ONE_RANK_GPU=0 run_cell s96q30t2b032lf q3-30b-a3b "asym_cpuadamwds|T2B" 32000 "1" "$POL" 1)
echo "FG-PROBE16 canary with LF@ebde34d3 -> $v (TRAINED => root cause = LF 7db8f687 x -46 asym fg path)" >> "$S"
echo "=== FG-PROBE16 DONE $(date '+%F %H:%M:%S') ===" >> "$S"
