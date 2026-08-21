#!/bin/bash
# stdtp_host_run.sh — HOST side: exec a chain inside the container.
# usage: stdtp_host_run.sh <NVD e.g. "1,3" or "3"> <container-path-of-chain>
ST=/scratch_local/user_data/shutian/kevin/enroot
NVD="$1"; CHAIN="$2"
exec env ENROOT_CONFIG_PATH=$ST/config ENROOT_DATA_PATH=$ST/data \
  ENROOT_CACHE_PATH=$ST/cache ENROOT_RUNTIME_PATH=$ST/runtime/ ENROOT_TEMP_PATH=$ST/tmp \
  enroot start --rw --root --env NVIDIA_VISIBLE_DEVICES=$NVD \
  --mount=/home/kevinni/AsymGEMM-SFT-38:/workspace/AsymGEMM-SFT-38 \
  --mount=/home/kevinni/env:/workspace/env:rw,bind \
  --mount=/scratch_local/user_data/shutian/kevin/cache:/scratch_local/user_data/shutian/kevin/cache \
  asym_sft_46 /bin/bash -c "bash $CHAIN"
