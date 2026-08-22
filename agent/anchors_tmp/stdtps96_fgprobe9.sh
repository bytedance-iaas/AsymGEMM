#!/bin/bash
set -uo pipefail
. /workspace/AsymGEMM-SFT-46/third_party/AsymGEMM/agent/anchors_tmp/stdtps96_lib.sh
echo "=== FG-PROBE9 (gradoff FALSE — the -46 tree's banked fg protocol) BEGIN $(date '+%F %H:%M:%S') ===" >> "$S"
occupiers_alive() { return 0; }
guard() { return 0; }
POL="none|false|false|false|false|false"
HYOFF="ASYM_OFFLOAD_MODULES=routed_experts,shared_experts,attention,norms,mlp_dense"
v=$(ONE_RANK_GPU=0 ARM_ENV="$HYOFF ASYM_CPU_ADAMW_GRAD_OFFLOAD=false ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=false" run_cell s96h1t2b064gf hunyuan-a13b "asym_cpuadamwds|T2B" 64000 "2" "$POL" 1)
echo "FG-PROBE9 GPU0 no-occ HYOFF + gradofffalse -> $v" >> "$S"
echo "=== FG-PROBE9 DONE $(date '+%F %H:%M:%S') ===" >> "$S"
