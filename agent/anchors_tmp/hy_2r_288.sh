#!/bin/bash
# 2-rank 288k tiebreak pair: uns (expect GOOM from 98%@256k) vs asym sdp2-T1
# (89%@256k -> should fit) — decides the 2-rank sole-survivor rung.
set -uo pipefail
export GPU="0,1" HOSTFLOOR=1100
source /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib_c17.sh
S2="/workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/hy_status.log"
export CUDA_VISIBLE_DEVICES="0,1" GPU_POOL="0,1"
export ASYM_OFFLOAD_MODULES="routed_experts,shared_experts,attention,norms,mlp_dense"
POL="none|false|false|false|false|false"
cell() { local v; v=$(run_cell "$1" hunyuan-a13b "$2" "$3" "1" "$POL" 2); echo "HY CELL $1 r2 ${2%%|*} s=$3 -> $v $(date +%H:%M:%S)" >> "$S2"; echo "$v"; }
cell hy288_un "superoffload_mem|unsloth-ohbm0" 288000
cell hy288_asy "asym_sdp2_cpuadamwds|T1" 288000
echo "HY_288_DONE $(date +%H:%M:%S)" >> "$S2"
