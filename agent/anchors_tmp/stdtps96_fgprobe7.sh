#!/bin/bash
set -uo pipefail
. /workspace/AsymGEMM-SFT-46/third_party/AsymGEMM/agent/anchors_tmp/stdtps96_lib.sh
echo "=== FG-PROBE7 (fresh JIT cache) BEGIN $(date '+%F %H:%M:%S') ===" >> "$S"
occupiers_alive() { return 0; }
guard() { return 0; }
POL="none|false|false|false|false|false"
v=$(ONE_RANK_GPU=0 ARM_ENV="PYTHONFAULTHANDLER=1 DG_JIT_CACHE_DIR=/tmp/asym_jit_fresh96" run_cell s96h1t2b064g0j hunyuan-a13b "asym_cpuadamwds|T2B" 64000 "2" "$POL" 1)
echo "FG-PROBE7 GPU0 no-occ FRESH-JIT-CACHE -> $v (TRAINED => poisoned cache; segv => code regression, bisect)" >> "$S"
echo "=== FG-PROBE7 DONE $(date '+%F %H:%M:%S') ===" >> "$S"
