#!/bin/bash
# 1-rank 544k tiebreak pair: uo (proj ~197 GiB, near-certain GOOM) vs asym
# T2B (proj ~188, within slope error of 184 — coin flip). Breaks or tightens
# the 512k co-last-standing tie.
set -uo pipefail
export GPU=0 HOSTFLOOR=1100
source /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib_c17.sh
S2="/workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/hy_status.log"
export CUDA_VISIBLE_DEVICES=0 GPU_POOL=0
export ASYM_OFFLOAD_MODULES="routed_experts,shared_experts,attention,norms,mlp_dense"
POL="none|false|false|false|false|false"
rm -rf /dev/shm/asym_fabric_* 2>/dev/null || true
cell() { local v; v=$(run_cell "$1" hunyuan-a13b "$2" "$3" "1" "$POL" 1); echo "HY CELL $1 r1 ${2%%|*} s=$3 -> $v $(date +%H:%M:%S)" >> "$S2"; echo "$v"; }
cell hy544_uo "superoffload_mem|unsloth-off-ohbm0" 544000
cell hy544_a2b "asym_cpuadamwds|T2B" 544000
echo "HY_544_DONE $(date +%H:%M:%S)" >> "$S2"
