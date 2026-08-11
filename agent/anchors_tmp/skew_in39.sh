#!/bin/bash
# Non-interactive `asym39_enroot_run` twin for the ep-skew campaign (c11).
# Same container (asym_sft_39), same mounts as the CURRENT interactive helper
# (workspace = ${CUSTOM_ENROOT_WS}/AsymGEMM-SFT), only the final command differs.
#
#   usage: [GPUS=0,1] skew_in39.sh '<command>'
#   cwd inside the container: /workspace/AsymGEMM-SFT/third_party/AsymGEMM
set -uo pipefail
source ~/env/bashrc.sh >/dev/null 2>&1 || source ~/.bashrc >/dev/null 2>&1

name=asym_sft_39
host_folder="${CUSTOM_ENROOT_WS}/AsymGEMM-SFT"
workdir="/workspace/AsymGEMM-SFT"

_custom_enroot_prepare

mounts=(
  "--mount=${host_folder}:${workdir}"
  "--mount=${CUSTOM_ENROOT_WS}/env:/workspace/env:rw,bind"
  "--mount=${CUSTOM_ENROOT_CACHE}:${CUSTOM_ENROOT_CACHE}"
)

envs=(--env "HF_HOME=/scratch_local/user_data/shutian/kevin/cache/huggingface")
if [[ -n "${GPUS:-}" ]]; then
  # enroot/libnvidia-container renumbers the selected GPUs to 0..n-1 inside
  # the container: NVIDIA_VISIBLE_DEVICES picks the physical GPUs, and
  # CUDA_VISIBLE_DEVICES must use the RENUMBERED indices (passing the
  # physical ids through leaves torch with zero devices — measured).
  n=$(awk -F, '{print NF}' <<< "${GPUS}")
  cvd=$(seq -s, 0 $((n - 1)))
  envs+=(
    --env "CUDA_VISIBLE_DEVICES=${cvd}"
    --env "NVIDIA_VISIBLE_DEVICES=${GPUS}"
  )
fi

_custom_enroot_exec enroot start --rw --root "${mounts[@]}" "${envs[@]}" "${name}" \
  /bin/bash -lc "cd '${workdir}/third_party/AsymGEMM' && $*"
