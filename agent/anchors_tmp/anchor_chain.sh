#!/bin/bash
# Tier-ladder anchor runs for mixtral (user-approved, 2026-07-27):
# T1@64k·b2, T2@64k·b2, then T2@64k·b3 (or T2@64k·b1 if b2 GOOMs).
# Invocation pattern copied from the campaign's frozen scripts (mixtral_smoke3.sh / pair_walker.sh).
set -uo pipefail
cd /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM
export PROFILERS=source MAX_STEPS=2 WARMUP_STEPS=1 MAX_SAMPLES=512 DATASET_OVERWRITE=false OVERWRITE=false
export ASYM_ZERO_ROUTER_JITTER=1 TRUST_REMOTE_CODE=false
export HF_HOME=/scratch_local/user_data/shutian/kevin/cache/huggingface
export CUDA_VISIBLE_DEVICES=0
LOGD=/workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp
S="$LOGD/status.log"
B=profiling_results/profiling/asym_long_sft_smoke__lora__lf__bf16

guard() { for i in $(seq 1 60); do n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l); if [ "$n" -eq 0 ]; then rm -f /dev/shm/asym_fabric_* 2>/dev/null; return 0; fi; sleep 20; done; echo "GUARD-FAIL $(date +%H:%M)" >> "$S"; exit 9; }

verdict() { local dtag="$1" log="$2"
  local tsv="$B/${dtag}/jobs.tsv"
  if [ -f "$tsv" ] && awk -F'\t' 'NR>1 && ($1=="ok"||$1=="failed:1"){f=1} END{exit !f}' "$tsv"; then echo TRAINED; return; fi
  grep -aq "host-mem-watchdog" "$log" 2>/dev/null && { echo COOM; return; }
  grep -aqE "OutOfMemoryError|CUDA out of memory" "$log" 2>/dev/null && { echo GOOM; return; }
  echo FAIL
}

run_one() { local tag="$1" tier="$2" b="$3"; guard
  echo "START $tag (tier=$tier b=$b) $(date +%H:%M)" >> "$S"
  RUN_NAME="${tag}-c14_mixtral-8x22b" RUNS="mixtral-8x22b|1 ; asym_cpuadamwds|${tier}|ligerloss1 ; 64000|${b}|1 ; none|false|false|false|false|false" \
    bash scripts/lf/profile_lora_lf_test_source.sh >> "$LOGD/r_${tag}.log" 2>&1
  local v; v=$(verdict "${tag}-c14_mixtral-8x22b__b${b}_s64000_ga1_drop000" "$LOGD/r_${tag}.log")
  echo "RESULT $tag -> $v $(date +%H:%M)" >> "$S"
  echo "$v"
}

v1=$(run_one at1b2 T1 2)
v2=$(run_one at2b2 T2 2)
if [ "$v2" = "GOOM" ]; then v3=$(run_one at2b1 T2 1); else v3=$(run_one at2b3 T2 3); fi
echo "ANCHORS-DONE t1b2=$v1 t2b2=$v2 third=$v3 $(date +%H:%M)" >> "$S"
