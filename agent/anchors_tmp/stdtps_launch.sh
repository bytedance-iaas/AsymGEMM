#!/usr/bin/env bash
# stdtps_launch.sh — host-side launcher: run a chain script inside the
# asym_sft_39 enroot container (non-interactive; same container/mounts as
# asym39_enroot_run, command in place of the interactive shell).
# Usage: stdtps_launch.sh <in-container-script-path> <gpus e.g. "0" or "0,1">
script="${1:?usage: stdtps_launch.sh <script> <gpus>}"
gpus="${2:?usage: stdtps_launch.sh <script> <gpus>}"
source /home/kevinni/env/bashrc.sh >/dev/null 2>&1
if ! type _custom_enroot_exec >/dev/null 2>&1; then
  echo "FATAL: _custom_enroot_exec not defined after sourcing ~/env/bashrc.sh" >&2
  exit 1
fi
_custom_enroot_prepare
_custom_enroot_exec enroot start --rw --root \
  "--mount=/home/kevinni/AsymGEMM-SFT-39:/workspace/AsymGEMM-SFT-39" \
  "--mount=/home/kevinni/env:/workspace/env:rw,bind" \
  "--mount=${CUSTOM_ENROOT_CACHE}:${CUSTOM_ENROOT_CACHE}" \
  --env "CUDA_VISIBLE_DEVICES=${gpus}" --env "NVIDIA_VISIBLE_DEVICES=${gpus}" \
  asym_sft_39 /bin/bash -lc "bash ${script}"
