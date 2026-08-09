#!/bin/bash
# SO-recomp @ mixtral's nemo last-fit seq (2-rank, 8k), batch walk.
set -uo pipefail
cd /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM
export PROFILERS=source MAX_STEPS=2 WARMUP_STEPS=1 MAX_SAMPLES=512 DATASET_OVERWRITE=false OVERWRITE=false
export ASYM_ZERO_ROUTER_JITTER=1 TRUST_REMOTE_CODE=false
export HF_HOME=/scratch_local/user_data/shutian/kevin/cache/huggingface
export XDG_CACHE_HOME=/scratch_local/user_data/shutian/kevin/cache
S="agent/anchors_tmp/soeq_status.log"
B=profiling_results/profiling/asym_long_sft_smoke__lora__lf__bf16
for b in 8 4 2; do
  while [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^$/d' | wc -l)" -ne 0 ]; do sleep 20; done
  echo "START somxr2s8 r2 s=8000 b=$b $(date +%H:%M:%S)" >> "$S"
  RUN_NAME="somxr2s8_mixtral-8x22b" GPU_POOL="0,1" \
  RUNS="mixtral-8x22b|2 ; superoffload_mem|recomp|ligerloss1 ; 8000|$b|1 ; none|false|false|false|false|false" \
    bash scripts/lf/profile_lora_lf_test_source.sh >> "agent/anchors_tmp/soeq_somx_b${b}.log" 2>&1
  tsv="$B/somxr2s8_mixtral-8x22b__b${b}_s8000_ga1_drop000/jobs.tsv"
  v=FAIL
  grep -aqE "OutOfMemoryError|CUDA out of memory" "agent/anchors_tmp/soeq_somx_b${b}.log" && v=GOOM
  [ -f "$tsv" ] && awk -F'\t' 'NR>1 && $1=="ok"{f=1} END{exit !f}' "$tsv" && v=TRAINED
  echo "CELL somxr2s8 b=$b -> $v $(date +%H:%M:%S)" >> "$S"
  sleep 30
  [ "$v" = "TRAINED" ] && break
done
echo SOMX_DONE >> "$S"
