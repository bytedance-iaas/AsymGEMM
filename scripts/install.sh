#!/usr/bin/env bash
# Fresh-clone setup and editable install for AsymGEMM.

set -euo pipefail

# =============================================================================
# User Parameters
# =============================================================================
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0a}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DG_JIT_WITH_LINEINFO="${DG_JIT_WITH_LINEINFO:-1}"
export DG_JIT_CLEAR_CACHE="${DG_JIT_CLEAR_CACHE:-0}"

CUTLASS_REF="${CUTLASS_REF:-v4.5.0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# =============================================================================
# Derived Parameters
# =============================================================================
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# =============================================================================
# Main Logic
# =============================================================================
cd "${ROOT_DIR}"

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

  if git submodule update --init --recursive; then
    return
  fi

  echo "Recorded submodule checkout failed; falling back to CUTLASS ${CUTLASS_REF}."
  git -C third-party/cutlass fetch --tags origin
  git -C third-party/cutlass checkout --force "${CUTLASS_REF}"

  git -C third-party/fmt fetch origin
  git -C third-party/fmt checkout --force 4b50ad794422c6ecbf773141a09592fd9061a6fb \
    || git -C third-party/fmt checkout --force origin/master
}

install_package() {
  log "Installing Python requirements"
  require_cmd "${PYTHON_BIN}"
  "${PYTHON_BIN}" -m pip install -r requirements.txt

  log "Checking Python/CUDA environment"
  "${PYTHON_BIN}" - <<'PY'
import torch
from torch.utils.cpp_extension import CUDA_HOME

print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("CUDA_HOME:", CUDA_HOME)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY

  log "Installing asym_gemm editable"
  rm -rf build dist *.egg-info
  if [[ "${DG_JIT_CLEAR_CACHE}" == "1" ]]; then
    log "Clearing AsymGEMM JIT cache: ${HOME}/.asym_gemm/cache"
    rm -rf "${HOME}/.asym_gemm/cache"
  fi
  "${PYTHON_BIN}" -m pip install --no-build-isolation -e .

  "${PYTHON_BIN}" - <<'PY'
import asym_gemm

print("AsymGEMM", asym_gemm.__version__)
print("module:", asym_gemm.__file__)
PY
}

prepare_submodules
install_package

log "Done"
