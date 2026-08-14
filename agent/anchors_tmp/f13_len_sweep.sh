#!/bin/bash
# Share-vs-length sweep: real 2-step b1 anchor cells at 48k/64k/80k/100k for
# qwen3-30b(T2) / glm4.7-flash(T1) / q3.5-122b(T1), serial (node discipline).
# Runs inside asym_sft_39. Logs per cell; skips cells whose run dir exists.
set -uo pipefail
L=/scratch_local/user_data/shutian/kevin/cache/ep_skew_logs
export HF_HOME=/scratch_local/user_data/shutian/kevin/cache/huggingface
export NUMACTL_ENABLE=1 NUMACTL_MODE=membind NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1
export PROFILERS=source MAX_STEPS=2 WARMUP_STEPS=1 MAX_SAMPLES=512
export CUDA_VISIBLE_DEVICES=0,1 GPU_POOL=0,1 DDP_TIMEOUT=1500

cell() { # cell <modelkey> <tier> <seq> <tag> [extra_env]
  local mk="$1" tier="$2" seq="$3" tag="$4"
  local dir="profiling_results/profiling/asym_long_sft_smoke__lora__lf__bf16/${tag}__b1_s${seq}_ga1_drop000"
  if compgen -G "$dir/*/step_samples.json" >/dev/null 2>&1 || [ -d "$dir" ]; then
    echo "[skip] $tag exists"; return 0
  fi
  rm -f /dev/shm/asym_fabric_*
  echo "[cell] $tag start $(date +%H:%M:%S)"
  RUN_NAME="$tag" RUNS="${mk}|2 ; asym_sdp2_cpuadamwds|${tier}|ligerloss1 ; ${seq}|1|1 ; none|false|false|false|false|false" \
    bash scripts/lf/profile_lora_lf_test_source.sh > "$L/${tag}.log" 2>&1
  echo "[cell] $tag done rc=$? $(date +%H:%M:%S)"
}

for SEQ in 48000 64000 80000 100000; do
  T=$((SEQ/1000))
  cell q3-30b-a3b T2 "$SEQ" "fs${T}q30b"
  cell glm4.7-flash T1 "$SEQ" "fs${T}gflash"
  ASYM_ARENA_SHM_CAP_GB=400 cell q3.5-122b-a10b T1 "$SEQ" "fs${T}q122b"
done
echo "LEN_SWEEP_CELLS_DONE"
