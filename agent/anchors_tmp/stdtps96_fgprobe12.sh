#!/bin/bash
set -uo pipefail
. /workspace/AsymGEMM-SFT-46/third_party/AsymGEMM/agent/anchors_tmp/stdtps96_lib.sh
echo "=== FG-PROBE12 (qwen canary on GPU1 — never occupied) BEGIN $(date '+%F %H:%M:%S') ===" >> "$S"
occupiers_alive() { return 0; }
guard() { return 0; }
POL="none|false|false|false|false|false"
v=$(ONE_RANK_GPU=1 ARM_ENV="PYTHONFAULTHANDLER=1" run_cell s96q30t2b032g1 q3-30b-a3b "asym_cpuadamwds|T2B" 32000 "1" "$POL" 1)
echo "FG-PROBE12 qwen30B T2B@32k b1 GPU1 -> $v (TRAINED => GPUs 0/2 wedged; segv => node-wide beyond GPUs)" >> "$S"
echo "=== FG-PROBE12 DONE $(date '+%F %H:%M:%S') ===" >> "$S"
