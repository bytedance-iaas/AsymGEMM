#!/usr/bin/env bash
script="${1:?}"; gpus="${2:-none}"
source /home/kevinni/env/bashrc.sh >/dev/null 2>&1
_custom_enroot_prepare
_custom_enroot_exec enroot start --rw --root \
  "--mount=/home/kevinni/AsymGEMM-SFT-46:/workspace/AsymGEMM-SFT-46" \
  "--mount=/home/kevinni/AsymGEMM-SFT-39:/workspace/AsymGEMM-SFT-39" \
  "--mount=/home/kevinni/env:/workspace/env:rw,bind" \
  "--mount=${CUSTOM_ENROOT_CACHE}:${CUSTOM_ENROOT_CACHE}" \
  --env "CUDA_VISIBLE_DEVICES=${gpus}" --env "NVIDIA_VISIBLE_DEVICES=${gpus}" \
  asym_sft_46 /bin/bash -lc "bash ${script}"
