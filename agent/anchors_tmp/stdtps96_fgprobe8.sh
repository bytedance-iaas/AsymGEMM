#!/bin/bash
set -uo pipefail
. /workspace/AsymGEMM-SFT-46/third_party/AsymGEMM/agent/anchors_tmp/stdtps96_lib.sh
echo "=== FG-PROBE8 (HEAD + hunyuan family offload list) BEGIN $(date '+%F %H:%M:%S') ===" >> "$S"
occupiers_alive() { return 0; }
guard() { return 0; }
POL="none|false|false|false|false|false"
v=$(ONE_RANK_GPU=0 ARM_ENV="ASYM_OFFLOAD_MODULES=routed_experts,shared_experts,attention,norms,mlp_dense" run_cell s96h1t2b064hf hunyuan-a13b "asym_cpuadamwds|T2B" 64000 "2" "$POL" 1)
echo "FG-PROBE8 GPU0 no-occ HEAD + family ASYM_OFFLOAD_MODULES -> $v (TRAINED => incident closed: protocol, not regression)" >> "$S"
echo "=== FG-PROBE8 DONE $(date '+%F %H:%M:%S') ===" >> "$S"
