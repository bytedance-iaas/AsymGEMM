#!/bin/bash
set -uo pipefail
cd /workspace/AsymGEMM-SFT-46/third_party/AsymGEMM
git checkout 473ef84 -- asym_gemm/ csrc/ scripts/lf/   # restore HEAD after the bisect control
. agent/anchors_tmp/stdtps96_lib.sh
echo "=== FG-PROBE11 (qwen fg canary) BEGIN $(date '+%F %H:%M:%S') ===" >> "$S"
occupiers_alive() { return 0; }
guard() { return 0; }
POL="none|false|false|false|false|false"
v=$(ONE_RANK_GPU=0 ARM_ENV="PYTHONFAULTHANDLER=1" run_cell s96q30t2b032 q3-30b-a3b "asym_cpuadamwds|T2B" 32000 "1" "$POL" 1)
echo "FG-PROBE11 qwen30B 1r T2B@32k b1 GPU0 -> $v (segv => NODE-WIDE fg breakage; TRAINED => hunyuan-specific)" >> "$S"
echo "=== FG-PROBE11 DONE $(date '+%F %H:%M:%S') ===" >> "$S"
