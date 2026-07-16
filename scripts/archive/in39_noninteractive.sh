#!/bin/bash
# Non-interactive equivalent of `asym39_enroot_run`, which ends in `exec bash -i`
# and so cannot take a command. Same container (asym_sft_39), same mounts, same
# env plumbing — only the final command differs. Nothing here runs on the host.
#
#   usage: [GPUS=0] in39.sh '<command>'
#   cwd inside the container: /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM
set -uo pipefail
source ~/env/bashrc.sh >/dev/null 2>&1

name=asym_sft_39
host_folder="${CUSTOM_ENROOT_WS}/AsymGEMM-SFT-39"
workdir="/workspace/AsymGEMM-SFT-39"

_custom_enroot_prepare

mounts=(
  "--mount=${host_folder}:${workdir}"
  "--mount=${CUSTOM_ENROOT_WS}/env:/workspace/env:rw,bind"
  "--mount=${CUSTOM_ENROOT_CACHE}:${CUSTOM_ENROOT_CACHE}"
)

envs=()
if [[ -n "${GPUS:-}" ]]; then
  envs=(
    --env "CUDA_VISIBLE_DEVICES=${GPUS}"
    --env "NVIDIA_VISIBLE_DEVICES=${GPUS}"
  )
fi

_custom_enroot_exec enroot start --rw --root "${mounts[@]}" "${envs[@]}" "${name}" \
  /bin/bash -lc "cd '${workdir}/third_party/AsymGEMM' && $*"
