#!/bin/bash
# Run 4: T3 @ 64k·b2 on CURRENT code (moefg unlocked) — the vl-t3 numbers are
# pre-moefg and not comparable to the fresh T1/T2 anchors. Waits for the main
# anchor chain (ANCHORS-DONE in status.log) before touching the GPU.
set -uo pipefail
cd /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM
export PROFILERS=source MAX_STEPS=2 WARMUP_STEPS=1 MAX_SAMPLES=512 DATASET_OVERWRITE=false OVERWRITE=false
export ASYM_ZERO_ROUTER_JITTER=1 TRUST_REMOTE_CODE=false
export HF_HOME=/scratch_local/user_data/shutian/kevin/cache/huggingface
export CUDA_VISIBLE_DEVICES=0
LOGD=/workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp
S="$LOGD/status.log"
B=profiling_results/profiling/asym_long_sft_smoke__lora__lf__bf16

for i in $(seq 1 240); do grep -q "ANCHORS-DONE" "$S" 2>/dev/null && break; sleep 15; done
grep -q "ANCHORS-DONE" "$S" || { echo "T3RERUN-ABORT chain never finished $(date +%H:%M)" >> "$S"; exit 1; }
for i in $(seq 1 60); do n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l); if [ "$n" -eq 0 ]; then rm -f /dev/shm/asym_fabric_* 2>/dev/null; break; fi; sleep 20; done

echo "START at3b2 (tier=T3-current b=2) $(date +%H:%M)" >> "$S"
RUN_NAME="at3b2-c14_mixtral-8x22b" RUNS="mixtral-8x22b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1 ; 64000|2|1 ; none|false|false|false|false|false" \
  bash scripts/lf/profile_lora_lf_test_source.sh >> "$LOGD/r_at3b2.log" 2>&1
tsv="$B/at3b2-c14_mixtral-8x22b__b2_s64000_ga1_drop000/jobs.tsv"
if [ -f "$tsv" ] && awk -F'\t' 'NR>1 && ($1=="ok"||$1=="failed:1"){f=1} END{exit !f}' "$tsv"; then v=TRAINED; else v=FAIL; grep -aqE "OutOfMemoryError|CUDA out of memory" "$LOGD/r_at3b2.log" && v=GOOM; fi
echo "RESULT at3b2 -> $v $(date +%H:%M)" >> "$S"
echo "T3RERUN-DONE $(date +%H:%M)" >> "$S"
