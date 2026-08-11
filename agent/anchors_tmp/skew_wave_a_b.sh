#!/bin/bash
# ep-skew campaign, waves for Agent A (qwen3-30b) then Agent B (glm4.7-flash,
# hunyuan-a13b) on c11. Runs AFTER Agent C frees the GPUs. Per-GPU replicas,
# datasets split by priority (P0 first within each replica list); the probe's
# manifest makes restarts idempotent.
set -uo pipefail
RUN=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/agent/anchors_tmp/skew_in39.sh
L=/scratch_local/user_data/shutian/kevin/cache/ep_skew_logs

wave() { # wave <model> <tag> <ds_gpu0> <ds_gpu1> <ds_gpu2> <ds_gpu3>
  local model="$1" tag="$2"; shift 2
  local pids=()
  local g=0
  for ds in "$@"; do
    if [[ -n "$ds" ]]; then
      GPUS=$g bash "$RUN" "python scripts/ep_skew/route_skew_probe.py --model $model --datasets $ds > $L/cell_${tag}_g${g}.log 2>&1; echo RC=\$?" &
      pids+=($!)
    fi
    g=$((g+1))
  done
  wait "${pids[@]}"
  echo "WAVE_${tag}_DONE"
}

# Agent A: qwen3-30b, 8 cells over 4 replicas (P0 cell first on each GPU)
wave qwen3-30b a30b "dapo,swebench" "codeforces,openscience" "sft_mix,longbench" "megamath,gpqa"

# Agent B wave 1: glm4.7-flash, same 8-cell grid
wave glm4.7-flash bflash "dapo,swebench" "codeforces,openscience" "sft_mix,longbench" "megamath,gpqa"

# Agent B wave 2: hunyuan-a13b, P1 triplet only (one cell per GPU, parallel)
wave hunyuan-a13b bhun "dapo" "codeforces" "sft_mix" ""

echo "ALL_WAVES_DONE"
