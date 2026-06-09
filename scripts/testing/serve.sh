#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-start}"
GPU_SPEC="${GPU:-${GPU_POOL:-${GPUS:-0}}}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${PORT:-${BASE_PORT:-30000}}"
HOST="${HOST:-0.0.0.0}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.8}"
SERVED_MODEL_PREFIX="${SERVED_MODEL_PREFIX:-gpu-test}"
RUN_DIR="${RUN_DIR:-/tmp/sglang-gpu-pool}"
SITE_PACKAGES="${SITE_PACKAGES:-/usr/local/lib/python3.12/dist-packages:/usr/lib/python3/dist-packages:/usr/lib/python3.12/dist-packages}"
SGLANG_SOURCE="${SGLANG_SOURCE:-/sgl-workspace/sglang/python}"
SGL_KERNEL_SOURCE="${SGL_KERNEL_SOURCE:-/sgl-workspace/sglang/sgl-kernel/python/sgl_kernel}"
SGL_KERNEL_WHEEL="${SGL_KERNEL_WHEEL:-/usr/local/lib/python3.12/dist-packages/sgl_kernel}"
SGL_KERNEL_OVERLAY="${SGL_KERNEL_OVERLAY:-${RUN_DIR}/sgl-kernel-overlay}"
SGLANG_PYTHONPATH="${SGLANG_SOURCE}:${SGL_KERNEL_OVERLAY}:${SITE_PACKAGES}${EXTRA_PYTHONPATH:+:${EXTRA_PYTHONPATH}}"

die() {
  echo "error: $*" >&2
  exit 2
}

usage() {
  echo "usage: $0 [start|serve]" >&2
  echo "env: GPU/GPU_POOL/GPUS selects one CUDA device, default 0" >&2
  echo "env: MODEL_PATH, PORT/BASE_PORT, HOST, MEM_FRACTION_STATIC tune SGLang" >&2
}

prepare_kernel_overlay() {
  if [[ -f "${SGL_KERNEL_OVERLAY}/sgl_kernel/__init__.py" ]] &&
    compgen -G "${SGL_KERNEL_OVERLAY}/sgl_kernel/sm100/common_ops*.so" >/dev/null; then
    return 0
  fi

  rm -rf "${SGL_KERNEL_OVERLAY}"
  mkdir -p "${SGL_KERNEL_OVERLAY}"
  cp -r --no-preserve=mode,ownership,timestamps \
    "${SGL_KERNEL_SOURCE}" \
    "${SGL_KERNEL_OVERLAY}/"

  if compgen -G "${SGL_KERNEL_WHEEL}/*.so" >/dev/null; then
    cp --no-preserve=mode,ownership,timestamps \
      "${SGL_KERNEL_WHEEL}"/*.so \
      "${SGL_KERNEL_OVERLAY}/sgl_kernel/"
  fi

  local arch
  for arch in sm90 sm100; do
    if compgen -G "${SGL_KERNEL_WHEEL}/${arch}/*.so" >/dev/null; then
      mkdir -p "${SGL_KERNEL_OVERLAY}/sgl_kernel/${arch}"
      cp --no-preserve=mode,ownership,timestamps \
        "${SGL_KERNEL_WHEEL}/${arch}"/*.so \
        "${SGL_KERNEL_OVERLAY}/sgl_kernel/${arch}/"
    fi
  done
}

case "${ACTION}" in
  start|serve)
    ;;
  *)
    usage
    exit 2
    ;;
esac

read -r -a GPU_IDS <<< "${GPU_SPEC//,/ }"
if (( ${#GPU_IDS[@]} != 1 )); then
  die "plain serve expects one GPU id, got '${GPU_SPEC}'"
fi

GPU_ID="${GPU_IDS[0]}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${SERVED_MODEL_PREFIX}-${GPU_ID}}"

prepare_kernel_overlay

echo "GPU ${GPU_ID}: starting plain SGLang foreground server on port ${PORT}"
echo "GPU ${GPU_ID}: press Ctrl-C to stop SGLang"

exec env \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  PYTHONPATH="${SGLANG_PYTHONPATH}" \
  python3 -S -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --load-format dummy \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --served-model-name "${SERVED_MODEL_NAME}"
