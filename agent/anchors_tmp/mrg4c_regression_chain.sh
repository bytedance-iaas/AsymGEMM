#!/bin/bash
# mrg4b (2026-08-14 4-way merge) regression chain: 4 anchored near-ceiling 1r
# cells on the merged tree (flavor-flipped 54acb99+), serial GPU0. Anchors:
#   q3-30b-a3b   asym|T2 @320k b1 -> 1370 tok/s (c17 mrg4 chain 2026-08-09; same node, tight band -3%)
#   glm4.7-flash asym|T3 @256k b1 ->  564 tok/s (c17 mrg4 chain 2026-08-09; trueT3 ker101/route101, tight -3%)
#   q3.5-35b-a3b asym|T2 @896k b1 -> 1492 tok/s (c18 fig-row campaign 07-23/24; cross-node band ~-8%)
#   glm4.5-air   asym|T1 @320k b1 ->  510 tok/s (c18 glmext 08-05, sole-survivor cell, RSS 878; cross-node band; arena 240 per run_glms; beyond-ctx RoPE cell)
# 1r cells never enable ASYM_EP_SEP -> the sepplanlink flavor flip is inert here;
# what they DO exercise: rebuilt _C, frozen_linear pre-gate-first+gates (disabled
# path), lf.py rotary component classify, dataset/driver merges.
set -uo pipefail
export GPU=0 HOSTFLOOR=1100
source /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib_c17.sh
S2="$LOGD/mrg4c_status.log"
POL="none|false|false|false|false|false"

shm_check() {
  if ! nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; then
    rm -f /dev/shm/asym_* 2>/dev/null || true
  fi
}

cell() { local tag="$1" model="$2" systok="$3" seq="$4" v
  shm_check
  v=$(run_cell "$tag" "$model" "$systok" "$seq" "1" "$POL" 1)
  echo "MRG4C $tag $model ${systok} s=$seq -> $v $(date +%H:%M:%S)" >> "$S2"
}

echo "MRG4C_CHAIN_START $(date +%H:%M:%S)" >> "$S2"
cell mrg4cq30 q3-30b-a3b "asym_cpuadamwds|T2" 320000
cell mrg4cgf glm4.7-flash "asym_cpuadamwds|T3" 256000
cell mrg4cq35 q3.5-35b-a3b "asym_cpuadamwds|T2" 896000
export ASYM_ARENA_SHM_CAP_GB=240
cell mrg4cair glm4.5-air "asym_cpuadamwds|T1" 320000
unset ASYM_ARENA_SHM_CAP_GB
echo "MRG4C_DONE $(date +%H:%M:%S)" >> "$S2"
