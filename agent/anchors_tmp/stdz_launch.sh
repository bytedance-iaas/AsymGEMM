#!/bin/bash
# stdz_launch.sh — host-side enroot launcher for standardization chains.
# Usage: stdz_launch.sh <GPUS(host ids, comma)> <inside_script_repo_relpath>
# Pattern copied from scripts/ep_skew/driver.sh (NVD filters+renumbers ->
# CVD must be inside ids 0..k-1).
set -u
GPUS="$1"; SCRIPT="$2"
N_GPU=$(awk -F, '{print NF}' <<<"$GPUS")
INSIDE=$(seq -s, 0 $((N_GPU-1)))
export ENROOT_CONFIG_PATH=/scratch_local/user_data/kevinni/enroot/config \
       ENROOT_DATA_PATH=/scratch_local/user_data/kevinni/enroot/data \
       ENROOT_CACHE_PATH=/scratch_local/user_data/kevinni/enroot/cache \
       ENROOT_RUNTIME_PATH=/scratch_local/user_data/kevinni/enroot/runtime/ \
       ENROOT_TEMP_PATH=/scratch_local/user_data/kevinni/enroot/tmp
exec enroot start --rw --root \
  --mount=/home/kevinni/AsymGEMM-SFT:/workspace/AsymGEMM-SFT \
  --mount=/home/kevinni/env:/workspace/env \
  --mount=/scratch_local/user_data/shutian/kevin/cache:/scratch_local/user_data/shutian/kevin/cache \
  --env NVIDIA_VISIBLE_DEVICES="$GPUS" --env CUDA_VISIBLE_DEVICES="$INSIDE" \
  ${OCC_PIDS:+--env OCC_PIDS="$OCC_PIDS"} \
  ${AIR_RESUME_CAP:+--env AIR_RESUME_CAP="$AIR_RESUME_CAP"} \
  ${AIR_RESUME_S:+--env AIR_RESUME_S="$AIR_RESUME_S"} \
  ${AIR_LADDER:+--env AIR_LADDER="$AIR_LADDER"} \
  ${AIR_DONE:+--env AIR_DONE="$AIR_DONE"} \
  ${FL_RESUME_LO:+--env FL_RESUME_LO="$FL_RESUME_LO"} \
  ${FL_LADDER:+--env FL_LADDER="$FL_LADDER"} \
  asym_sft_40 /bin/bash "/workspace/AsymGEMM-SFT/third_party/AsymGEMM/$SCRIPT"
