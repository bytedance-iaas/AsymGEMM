#!/bin/bash
# Max-coverage composition probe: selective recompute of the MoE block +
# fine-grained offload of the attention side. Does the wall move vs recomp?
set -uo pipefail
export XDG_CACHE_HOME=/scratch_local/user_data/shutian/kevin/cache
cd /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM
S="agent/anchors_tmp/nemo_q30b_status.log"
export NEMO_RECOMPUTE_MODULES=moe
export NEMO_OFFLOAD_MODULES=core_attn,attn_proj,qkv_linear,attn_norm,mlp_norm
run_cell() { # $1 tag $2 seq
  echo "START $1 selrecomp-actoff s=$2 $(date +%H:%M:%S)" >> "$S"
  RUNS="q3-30b-a3b|2 ; nemo|selrecomp-actoff|ligerloss1 ; $2|1|1 ; none|false|false|false|false|false" \
  RUN_NAME=nemo30b GPU_POOL=0,1 RUN_TIMEOUT_SECONDS=5400 OVERWRITE=false \
    bash scripts/lf/profile_lora_nemo.sh >> "agent/anchors_tmp/nemo30b_${1}.log" 2>&1
  v=$(grep -o 'VERDICT=[A-Z]*' "agent/anchors_tmp/nemo30b_${1}.log" | tail -1)
  echo "CELL $1 s=$2 -> ${v:-NONE} $(date +%H:%M:%S)" >> "$S"
  echo "${v#VERDICT=}"
}
run_cell sa128 128000
v=$(run_cell sa160 160000)
if [ "$v" = "TRAINED" ]; then
  run_cell sa192 192000
fi
echo SELAO_DONE >> "$S"
