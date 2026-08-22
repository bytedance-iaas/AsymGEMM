#!/bin/bash
set -uo pipefail
. /workspace/AsymGEMM-SFT-46/third_party/AsymGEMM/agent/anchors_tmp/stdtps96_lib.sh
echo "=== FG-PROBE15 (-46 code + -39 VENV) BEGIN $(date '+%F %H:%M:%S') ===" >> "$S"
occupiers_alive() { return 0; }
guard() { return 0; }
POL="none|false|false|false|false|false"
v=$(ONE_RANK_GPU=0 ARM_ENV="ENV_DIR=/workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/.venv" run_cell s96q30t2b032xv q3-30b-a3b "asym_cpuadamwds|T2B" 32000 "1" "$POL" 1)
echo "FG-PROBE15 -46 code + -39 venv -> $v (TRAINED => -46 VENV is the culprit)" >> "$S"
echo "=== FG-PROBE15 DONE $(date '+%F %H:%M:%S') ===" >> "$S"
