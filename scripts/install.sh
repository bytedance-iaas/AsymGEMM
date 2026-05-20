#!/bin/bash
# Fresh-clone setup and install path for AsymGEMM on H200/Hopper.
#
# This intentionally lives beside the original build/install scripts so those
# upstream files can stay easy to rebase.
#
# Defaults:
#   CUDA_HOME=/usr/local/cuda
#   TORCH_CUDA_ARCH_LIST=9.0a
#   CUDA_VISIBLE_DEVICES=0
#   DG_JIT_WITH_LINEINFO=1
#   DG_JIT_CLEAR_CACHE=0
#
# Useful overrides:
#   CUDA_VISIBLE_DEVICES=2 ./scripts/install.sh
#   CUTLASS_REF=v4.5.0 ./scripts/install.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0a}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DG_JIT_WITH_LINEINFO="${DG_JIT_WITH_LINEINFO:-1}"
export DG_JIT_CLEAR_CACHE="${DG_JIT_CLEAR_CACHE:-0}"

CUTLASS_REF="${CUTLASS_REF:-v4.5.0}"

log() {
  echo
  echo "==> $*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

prepare_submodules() {
  log "Preparing submodules"
  require_cmd git

  # The branch currently records a CUTLASS commit that may not be fetchable from
  # NVIDIA/cutlass. Try the recorded submodules first, then fall back to a known
  # CUTLASS release with the SM90/SM100 CuTe headers this branch includes.
  if git submodule update --init --recursive; then
    return
  fi

  echo "Recorded submodule checkout failed; falling back to CUTLASS ${CUTLASS_REF}."
  git -C third-party/cutlass fetch --tags origin
  git -C third-party/cutlass checkout --force "${CUTLASS_REF}"

  # fmt's recorded commit is fetchable in this workspace; keep trying the
  # recorded submodule path and only fall back to the default branch if needed.
  git -C third-party/fmt fetch origin
  git -C third-party/fmt checkout --force 4b50ad794422c6ecbf773141a09592fd9061a6fb \
    || git -C third-party/fmt checkout --force origin/master
}

install_package() {
  log "Installing asym_gemm"
  require_cmd python3
  require_cmd "${CUDA_HOME}/bin/nvcc"

  python3 - <<'PY'
import torch
from torch.utils.cpp_extension import CUDA_HOME

print("python/torch check")
print("  torch:", torch.__version__, "cuda:", torch.version.cuda)
print("  CUDA_HOME:", CUDA_HOME)
print("  cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  device:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY

  rm -rf build dist *.egg-info
  if [[ "${DG_JIT_CLEAR_CACHE}" == "1" ]]; then
    log "Clearing AsymGEMM JIT cache: ${HOME}/.asym_gemm/cache"
    rm -rf "${HOME}/.asym_gemm/cache"
  fi
  bash install.sh
}

prepare_submodules
install_package

log "Done"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT"

python -m pip install -r requirements.txt

rm -rf build dist asym_gemm.egg-info
python -m pip install --no-build-isolation -e .

python -c "import asym_gemm; print('AsymGEMM', asym_gemm.__version__)"
