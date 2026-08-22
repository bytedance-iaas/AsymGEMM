#!/bin/bash
# stdtps96_bisect.sh <commit> — path-scoped bisect step for the hunyuan-fg
# segfault (STDTPS96_c18.md incident 4→5): checkout asym_gemm/ csrc/ scripts/lf/
# at <commit>, run the 7-min repro (hy 1r T2B@64k b2, GPU0, no occupier), report.
set -uo pipefail
C="${1:?usage: stdtps96_bisect.sh <commit>}"
cd /workspace/AsymGEMM-SFT-46/third_party/AsymGEMM
git checkout "$C" -- asym_gemm/ csrc/ scripts/lf/ || exit 1
. agent/anchors_tmp/stdtps96_lib.sh
echo "=== BISECT $C BEGIN $(date '+%F %H:%M:%S') ===" >> "$S"
occupiers_alive() { return 0; }
guard() { return 0; }
POL="none|false|false|false|false|false"
v=$(ONE_RANK_GPU=0 ARM_ENV="PYTHONFAULTHANDLER=1" run_cell "s96bis${C:0:7}" hunyuan-a13b "asym_cpuadamwds|T2B" 64000 "2" "$POL" 1)
echo "BISECT $C -> $v" >> "$S"
echo "=== BISECT $C DONE $(date '+%F %H:%M:%S') ===" >> "$S"
