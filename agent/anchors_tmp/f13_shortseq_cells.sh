#!/bin/bash
# Short-seq deep-batch anchor cells: tokens/step held at 256k (2 ranks) so the
# walls128_* benches (same launch size) compose directly. Serial, b1-node rules.
set -uo pipefail
L=/scratch_local/user_data/shutian/kevin/cache/ep_skew_logs
export HF_HOME=/scratch_local/user_data/shutian/kevin/cache/huggingface
export NUMACTL_ENABLE=1 NUMACTL_MODE=membind NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1
export PROFILERS=source MAX_STEPS=2 WARMUP_STEPS=1 MAX_SAMPLES=512
export CUDA_VISIBLE_DEVICES=0,1 GPU_POOL=0,1 DDP_TIMEOUT=1500

cell() { # cell <modelkey> <tier> <seq> <batch> <tag>
  local mk="$1" tier="$2" seq="$3" b="$4" tag="$5"
  local dir="profiling_results/profiling/asym_long_sft_smoke__lora__lf__bf16/${tag}__b${b}_s${seq}_ga1_drop000"
  [ -d "$dir" ] && { echo "[skip] $tag exists"; return 0; }
  rm -f /dev/shm/asym_fabric_*
  echo "[cell] $tag start $(date +%H:%M:%S)"
  RUN_NAME="$tag" RUNS="${mk}|2 ; asym_sdp2_cpuadamwds|${tier}|ligerloss1 ; ${seq}|${b}|1 ; none|false|false|false|false|false" \
    bash scripts/lf/profile_lora_lf_test_source.sh > "$L/${tag}.log" 2>&1
  echo "[cell] $tag done rc=$? $(date +%H:%M:%S)"
}

cell q3-30b-a3b   T2 32000 4  fs32b4q30b
cell q3-30b-a3b   T2 16000 8  fs16b8q30b
cell q3-30b-a3b   T2 8000  16 fs8b16q30b
cell glm4.7-flash T1 32000 4  fs32b4gflash
cell glm4.7-flash T1 16000 8  fs16b8gflash
cell glm4.7-flash T1 8000  16 fs8b16gflash
ASYM_ARENA_SHM_CAP_GB=400 cell q3.5-122b-a10b T1 16000 8 fs16b8q122b
echo "SHORTSEQ_CELLS_DONE"
