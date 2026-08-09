#!/bin/bash
# SO-recomp cells at NeMo's last-fit seqs (same-seq comparison column).
# House protocol (tpfig_lib pattern): w1+m2, batch walk high->low, first
# TRAINED wins, serial, superoffload_mem|recomp|ligerloss1 on the LF stack.
set -uo pipefail
cd /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM
export PROFILERS=source MAX_STEPS=2 WARMUP_STEPS=1 MAX_SAMPLES=512 DATASET_OVERWRITE=false OVERWRITE=false
export ASYM_ZERO_ROUTER_JITTER=1 TRUST_REMOTE_CODE=false
export HF_HOME=/scratch_local/user_data/shutian/kevin/cache/huggingface
export XDG_CACHE_HOME=/scratch_local/user_data/shutian/kevin/cache
S="agent/anchors_tmp/soeq_status.log"
B=profiling_results/profiling/asym_long_sft_smoke__lora__lf__bf16

verdict() { local dtag="$1" log="$2"
  grep -aqE "OutOfMemoryError|CUDA out of memory" "$log" 2>/dev/null && { echo GOOM; return; }
  grep -aq "dropped below floor" "$log" 2>/dev/null && { echo COOM; return; }
  local tsv="$B/${dtag}/jobs.tsv"
  if [ -f "$tsv" ] && awk -F'\t' 'NR>1 && $1=="ok"{f=1} END{exit !f}' "$tsv"; then echo TRAINED; return; fi
  echo FAIL
}

run_cell() { # $1 tag $2 model $3 ranks $4 seq $5 "blist"
  local tag="$1" model="$2" ranks="$3" seq="$4" blist="$5" v=SKIP b gpus="0"
  [ "$ranks" = "2" ] && gpus="0,1"
  for b in $blist; do
    while [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^$/d' | wc -l)" -ne 0 ]; do sleep 20; done
    echo "START $tag $model r$ranks s=$seq b=$b $(date +%H:%M:%S)" >> "$S"
    RUN_NAME="${tag}_${model}" GPU_POOL="$gpus" \
    RUNS="${model}|${ranks} ; superoffload_mem|recomp|ligerloss1 ; ${seq}|${b}|1 ; none|false|false|false|false|false" \
      bash scripts/lf/profile_lora_lf_test_source.sh >> "agent/anchors_tmp/soeq_${tag}_b${b}.log" 2>&1
    local dmodel=${model//./_}
    v=$(verdict "${tag}_${dmodel}__b${b}_s${seq}_ga1_drop000" "agent/anchors_tmp/soeq_${tag}_b${b}.log")
    echo "CELL $tag r$ranks s=$seq b=$b -> $v $(date +%H:%M:%S)" >> "$S"
    sleep 30
    [ "$v" = "TRAINED" ] && break
    [ "$v" = "FAIL" ] && break
  done; echo "$v"; }

# 2-rank same-seq cells
run_cell so30r2s128 q3-30b-a3b 2 128000 "2 1"
run_cell so35r2s16 q3.5-35b-a3b 2 16000 "8 4 2"
run_cell so47r2s16 glm4.7-flash 2 16000 "16 8 4"
# 1-rank same-seq cells
run_cell so30r1s96 q3-30b-a3b 1 96000 "2 1"
run_cell so35r1s16 q3.5-35b-a3b 1 16000 "8 4 2"
run_cell so47r1s16 glm4.7-flash 1 16000 "8 4 2"
echo SOEQ_DONE >> "$S"
