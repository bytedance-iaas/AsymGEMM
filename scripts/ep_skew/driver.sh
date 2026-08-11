#!/bin/bash
# HOST-side replica launcher for the EP-skew screen (c12).
# Usage: driver.sh MODEL_KEY GPUS DS1[,DS2,...] [EXTRA_ARGS...]
# One enroot one-shot, pinned to GPUS. The probe itself skips manifest-done
# cells, so re-running after a crash is idempotent.
set -uo pipefail
MODEL=$1; GPUS=$2; DATASETS=$3; shift 3 || true
R=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
LOGD=$R/profiling_results/ep_skew/logs
mkdir -p "$LOGD"
LOGF=$LOGD/${MODEL}__gpu${GPUS//,/-}.log
# enroot's nvidia hook FILTERS to NVIDIA_VISIBLE_DEVICES and renumbers them
# 0..k-1 inside the container, so CUDA_VISIBLE_DEVICES must use inside ids
# (measured on c12: NVD=1 + CVD=1 -> zero visible devices -> CPU fallback).
N_GPU=$(awk -F, '{print NF}' <<<"$GPUS")
INSIDE=$(seq -s, 0 $((N_GPU-1)))

export ENROOT_CONFIG_PATH=/scratch_local/user_data/kevinni/enroot/config \
       ENROOT_DATA_PATH=/scratch_local/user_data/kevinni/enroot/data \
       ENROOT_CACHE_PATH=/scratch_local/user_data/kevinni/enroot/cache \
       ENROOT_RUNTIME_PATH=/scratch_local/user_data/kevinni/enroot/runtime/ \
       ENROOT_TEMP_PATH=/scratch_local/user_data/kevinni/enroot/tmp

echo "[driver] $MODEL gpus=$GPUS datasets=$DATASETS -> $LOGF"
enroot start --rw --root \
  --mount=/home/kevinni/AsymGEMM-SFT:/workspace/AsymGEMM-SFT \
  --mount=/home/kevinni/env:/workspace/env \
  --mount=/scratch_local/user_data/shutian/kevin/cache:/scratch_local/user_data/shutian/kevin/cache \
  --env NVIDIA_VISIBLE_DEVICES="$GPUS" --env CUDA_VISIBLE_DEVICES="$INSIDE" \
  asym_sft_40 /bin/bash /workspace/AsymGEMM-SFT/third_party/AsymGEMM/scripts/ep_skew/run_cell.sh \
  "$MODEL" "$DATASETS" "$@" >>"$LOGF" 2>&1
rc=$?
echo "[driver] $MODEL gpus=$GPUS rc=$rc"
exit $rc
