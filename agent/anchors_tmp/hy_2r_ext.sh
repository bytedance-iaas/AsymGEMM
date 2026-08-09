#!/bin/bash
# Hunyuan 2-rank EXTENSION — runs after HY_1R_EXT_DONE. asym deep tiers
# (sdp2|T2B, T3 fallback) from 320k until the wall, for the 2r last-standing arc.
set -uo pipefail
S2="/workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/hy_status.log"
until grep -q 'HY_1R_EXT_DONE' "$S2" 2>/dev/null; do sleep 120; done

export GPU="0,1" HOSTFLOOR=1100
source /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib_c17.sh
export CUDA_VISIBLE_DEVICES="0,1" GPU_POOL="0,1"
export ASYM_OFFLOAD_MODULES="routed_experts,shared_experts,attention,norms,mlp_dense"
POL="none|false|false|false|false|false"

cell() { local v; v=$(run_cell "$1" hunyuan-a13b "$2" "$3" "1" "$POL" 2); echo "HY CELL $1 r2 ${2%%|*} s=$3 -> $v $(date +%H:%M:%S)" >> "$S2"; echo "$v"; }

for s in 320000 384000 448000; do
  v=$(cell "hyx2_a2b$((s/1000))" "asym_sdp2_cpuadamwds|T2B" "$s")
  if [ "$v" != "TRAINED" ]; then
    v=$(cell "hyx2_a3t$((s/1000))" "asym_sdp2_cpuadamwds|T3" "$s")
    if [ "$v" != "TRAINED" ]; then
      echo "HY r2 ASYM-deep wall at $s" >> "$S2"
      break
    fi
  fi
done
echo "HY_2R_EXT_DONE $(date +%H:%M:%S)" >> "$S2"
