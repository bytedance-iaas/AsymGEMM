#!/bin/bash
# stdtps96_bisect2.sh <commit> — TRUE-repro bisect for the hunyuan-fg segfault:
# the d2sdp64 replica (2r sdp2 T2B@64k b2, GPUs 0,1, HYOFF, arena 320) which
# TRAINED on this tree 2026-08-11 and FAILS at HEAD.
set -uo pipefail
C="${1:?usage: stdtps96_bisect2.sh <commit>}"
cd /workspace/AsymGEMM-SFT-46/third_party/AsymGEMM
git checkout "$C" -- asym_gemm/ csrc/ scripts/lf/ || exit 1
. agent/anchors_tmp/stdtps96_lib.sh
echo "=== BISECT2 $C BEGIN $(date '+%F %H:%M:%S') ===" >> "$S"
occupiers_alive() { return 0; }
guard() { return 0; }
POL="none|false|false|false|false|false"
HYOFF="ASYM_OFFLOAD_MODULES=routed_experts,shared_experts,attention,norms,mlp_dense"
SIM_GPUS="0,1"
v=$(ARM_ENV="$HYOFF ASYM_ARENA_SHM_CAP_GB=320" run_cell "s96b2${C:0:7}" hunyuan-a13b "asym_sdp2_cpuadamwds|T2B" 64000 "2" "$POL" 2)
echo "BISECT2 $C -> $v" >> "$S"
echo "=== BISECT2 $C DONE $(date '+%F %H:%M:%S') ===" >> "$S"
