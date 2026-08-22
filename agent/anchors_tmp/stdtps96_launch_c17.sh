#!/bin/bash
# stdtps96_launch_c17.sh — host-side enroot launcher, c17 lane (Session E).
# Pattern = stdz_launch.sh (c12) with c17's node-local store + asym_sft_38
# (the container the parent tpfig campaign proved on this node).
# Usage: stdtps96_launch_c17.sh <GPUS(host ids, comma)> <inside_script_repo_relpath>
# NVD filters+renumbers -> CVD must be inside ids 0..k-1.
set -u
GPUS="$1"; SCRIPT="$2"
N_GPU=$(awk -F, '{print NF}' <<<"$GPUS")
INSIDE=$(seq -s, 0 $((N_GPU-1)))
export ENROOT_CONFIG_PATH=/scratch_local/user_data/shutian/kevin/enroot/config \
       ENROOT_DATA_PATH=/scratch_local/user_data/shutian/kevin/enroot/data \
       ENROOT_CACHE_PATH=/scratch_local/user_data/shutian/kevin/enroot/cache \
       ENROOT_RUNTIME_PATH=/scratch_local/user_data/shutian/kevin/enroot/runtime/ \
       ENROOT_TEMP_PATH=/scratch_local/user_data/shutian/kevin/enroot/tmp
exec enroot start --rw --root \
  --mount=/home/kevinni/AsymGEMM-SFT:/workspace/AsymGEMM-SFT \
  --mount=/home/kevinni/env:/workspace/env \
  --mount=/scratch_local/user_data/shutian/kevin/cache:/scratch_local/user_data/shutian/kevin/cache \
  --env NVIDIA_VISIBLE_DEVICES="$GPUS" --env CUDA_VISIBLE_DEVICES="$INSIDE" \
  ${CELL_TIMEOUT_MIN:+--env CELL_TIMEOUT_MIN="$CELL_TIMEOUT_MIN"} \
  ${HY2_RESUME_S:+--env HY2_RESUME_S="$HY2_RESUME_S"} \
  ${HY2_LADDER:+--env HY2_LADDER="$HY2_LADDER"} \
  asym_sft_38 /bin/bash "/workspace/AsymGEMM-SFT/third_party/AsymGEMM/$SCRIPT"
