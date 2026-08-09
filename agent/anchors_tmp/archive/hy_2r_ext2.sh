#!/bin/bash
# Hunyuan 2-rank deep-tier retry with the arena cap raised (the 320k T2B FAIL
# was 'shared fabric cap exceeded' at the 160-GiB default — hunyuan banks
# ~160 GB; GLM precedent: ASYM_ARENA_SHM_CAP_GB=240).
set -uo pipefail
export GPU="0,1" HOSTFLOOR=1100
source /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib_c17.sh
S2="/workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/hy_status.log"
export CUDA_VISIBLE_DEVICES="0,1" GPU_POOL="0,1"
export ASYM_OFFLOAD_MODULES="routed_experts,shared_experts,attention,norms,mlp_dense"
export ASYM_ARENA_SHM_CAP_GB=320
POL="none|false|false|false|false|false"
cell() { local v; v=$(run_cell "$1" hunyuan-a13b "$2" "$3" "1" "$POL" 2); echo "HY CELL $1 r2 ${2%%|*} s=$3 -> $v $(date +%H:%M:%S)" >> "$S2"; echo "$v"; }
for s in 320000 384000 448000; do
  v=$(cell "hyx4_a2b$((s/1000))" "asym_sdp2_cpuadamwds|T2B" "$s")
  [ "$v" = "TRAINED" ] || { echo "HY r2 ASYM-T2B wall at $s ($v) [arena320]" >> "$S2"; break; }
done
echo "HY_2R_EXT3_DONE $(date +%H:%M:%S)" >> "$S2"
