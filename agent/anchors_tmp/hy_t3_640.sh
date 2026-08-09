#!/bin/bash
# Close the 1-rank T3 bracket: 640k probe (608k ran at 185/189 GiB).
set -uo pipefail
export GPU=0 HOSTFLOOR=1100
source /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib_c17.sh
cd /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM
export CUDA_VISIBLE_DEVICES=0 GPU_POOL=0
export ASYM_OFFLOAD_MODULES="routed_experts,shared_experts,attention,norms,mlp_dense"
POL="none|false|false|false|false|false"
v=$(run_cell t3c640 hunyuan-a13b "asym_cpuadamwds|T3" 640000 "1" "$POL" 1)
echo "T3 CELL t3c640 r1 s=640000 -> $v $(date +%H:%M:%S)" >> agent/anchors_tmp/tpfig_status.log
echo "T3_640_DONE" >> agent/anchors_tmp/tpfig_status.log
