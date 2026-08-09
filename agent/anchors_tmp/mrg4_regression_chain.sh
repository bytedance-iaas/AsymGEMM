#!/bin/bash
# 4way-merge regression chain (2026-08-09): 3 anchored near-ceiling 1r cells
# on the merged tree, serial GPU0. Anchors:
#   q3-30b-a3b   asym|T2  @320k b1 -> 1336 tok/s (c14 figure DATA, tier-audit label T2)
#   hunyuan-a13b asym|T1  @256k b1 ->  929 tok/s (c17 HY_CAMPAIGN 1r table)
#   glm4.7-flash asym|T3  @256k b1 ->  563 tok/s (c18 fix_glm_t3 v2 smoke 08-09)
set -uo pipefail
export GPU=0 HOSTFLOOR=1100
source /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib_c17.sh
S2="$LOGD/mrg4_status.log"
POL="none|false|false|false|false|false"

shm_check() {
  # stale fabric arenas poison later launches (HY ops lesson) — clean when quiet
  if ! nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; then
    rm -f /dev/shm/asym_* 2>/dev/null || true
  fi
}

cell() { local tag="$1" model="$2" systok="$3" seq="$4" v
  shm_check
  v=$(run_cell "$tag" "$model" "$systok" "$seq" "1" "$POL" 1)
  echo "MRG4 $tag $model ${systok} s=$seq -> $v $(date +%H:%M:%S)" >> "$S2"
}

echo "MRG4_CHAIN_START $(date +%H:%M:%S)" >> "$S2"
cell mrg4q30 q3-30b-a3b "asym_cpuadamwds|T2" 320000
export ASYM_OFFLOAD_MODULES="routed_experts,shared_experts,attention,norms,mlp_dense"
cell mrg4hy hunyuan-a13b "asym_cpuadamwds|T1" 256000
unset ASYM_OFFLOAD_MODULES
cell mrg4gf glm4.7-flash "asym_cpuadamwds|T3" 256000
echo "MRG4_DONE $(date +%H:%M:%S)" >> "$S2"
