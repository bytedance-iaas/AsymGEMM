#!/bin/bash
set -uo pipefail
cd /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM
export PROFILERS=source MAX_STEPS=2 WARMUP_STEPS=1 MAX_SAMPLES=512 DATASET_OVERWRITE=false OVERWRITE=false
export ASYM_ZERO_ROUTER_JITTER=1 TRUST_REMOTE_CODE=false
export HF_HOME=/scratch_local/user_data/shutian/kevin/cache/huggingface
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN=""
export CUDA_VISIBLE_DEVICES=0 GPU_POOL=0
S=/workspace/AsymGEMM-SFT-46/third_party/AsymGEMM/agent/anchors_tmp/stdtps96_status.log
echo "=== FG-PROBE14 (-39 TREE canary on c18) BEGIN $(date '+%F %H:%M:%S') ===" >> "$S"
RUN_NAME="s96x39canary-c18_q3-30b-a3b" RUNS="q3-30b-a3b|1 ; asym_cpuadamwds|T2B|ligerloss1 ; 32000|1|1 ; none|false|false|false|false|false" \
  timeout -k 60 5400 bash scripts/lf/profile_lora_lf_test_source.sh > /workspace/AsymGEMM-SFT-46/third_party/AsymGEMM/agent/anchors_tmp/r_s96x39canary.log 2>&1
rc=$?
d=profiling_results/profiling/asym_long_sft_smoke__lora__lf__bf16/s96x39canary-c18_q3-30b-a3b__b1_s32000_ga1_drop000
ok=no; [ -f "$d/jobs.tsv" ] && awk -F'\t' 'NR>1 && $1=="ok"{f=1} END{exit !f}' "$d/jobs.tsv" && ok=yes
echo "FG-PROBE14 -39-tree canary rc=$rc jobs_ok=$ok (yes => -46 env drifted; no+segv => container/node-wide)" >> "$S"
echo "=== FG-PROBE14 DONE $(date '+%F %H:%M:%S') ===" >> "$S"
