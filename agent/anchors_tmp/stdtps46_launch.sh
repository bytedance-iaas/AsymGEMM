#!/usr/bin/env bash
# stdtps46_launch.sh — host-side launcher for the c18 / SFT-46 lane of the
# TP-figure x-axis standardization campaign (agent/impls/s04-p1-dgx-02-c06/
# standardize_tps.md). Runs a script INSIDE the asym_sft_46 enroot container,
# non-interactively, with the same mounts/env as asym46_enroot_run.
# Usage: stdtps46_launch.sh <in-container-script-path> <gpus e.g. "0" or "0,1">
script="${1:?usage: stdtps46_launch.sh <script> <gpus>}"
gpus="${2:-none}"   # "none" hides all GPUs (download/CPU-only jobs)
source /home/kevinni/env/bashrc.sh >/dev/null 2>&1
if ! type _custom_enroot_exec >/dev/null 2>&1; then
  echo "FATAL: _custom_enroot_exec not defined after sourcing ~/env/bashrc.sh" >&2
  exit 1
fi
_custom_enroot_prepare
_custom_enroot_exec enroot start --rw --root \
  "--mount=/home/kevinni/AsymGEMM-SFT-46:/workspace/AsymGEMM-SFT-46" \
  "--mount=/home/kevinni/env:/workspace/env:rw,bind" \
  "--mount=${CUSTOM_ENROOT_CACHE}:${CUSTOM_ENROOT_CACHE}" \
  --env "CUDA_VISIBLE_DEVICES=${gpus}" --env "NVIDIA_VISIBLE_DEVICES=${gpus}" \
  asym_sft_46 /bin/bash -lc "bash ${script}"
