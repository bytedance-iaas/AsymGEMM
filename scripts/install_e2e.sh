#!/bin/bash
# Fresh-clone setup, install, and validation path for AsymGEMM on H200/Hopper.
#
# This intentionally lives beside the original build/install scripts so those
# upstream files can stay easy to rebase.
#
# Defaults:
#   CUDA_HOME=/usr/local/cuda
#   TORCH_CUDA_ARCH_LIST=9.0a
#   CUDA_VISIBLE_DEVICES=0
#
# Useful overrides:
#   RUN_TESTS=0 ./scripts/install_e2e.sh
#   CUDA_VISIBLE_DEVICES=2 ./scripts/install_e2e.sh
#   CUTLASS_REF=v4.5.0 ./scripts/install_e2e.sh
#   TEST_FILES="tests/test_h20_bf16.py tests/test_h20_fp8.py" ./scripts/install_e2e.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0a}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RUN_TESTS="${RUN_TESTS:-1}"
CUTLASS_REF="${CUTLASS_REF:-v4.5.0}"
TEST_FILES="${TEST_FILES:-tests/test_h20_bf16.py tests/test_h20_fp8.py}"

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
  bash install.sh
}

validate_install() {
  log "Validating installed package import from /tmp"
  (
    cd /tmp
    python3 - <<'PY'
import asym_gemm
import asym_gemm.utils

print("asym_gemm:", asym_gemm.__file__)
print("version:", asym_gemm.__version__)
required = [
    "m_grouped_fp8_asym_gemm_nt_contiguous",
    "m_grouped_fp8_asym_gemm_nt_masked",
    "m_grouped_bf16_asym_gemm_nt_contiguous",
    "m_grouped_bf16_asym_gemm_nt_masked",
]
missing = [name for name in required if not hasattr(asym_gemm, name)]
if missing:
    raise RuntimeError(f"missing exported kernels: {missing}")
print("exported kernel check: PASS")
PY
  )

  log "Running existing SM90/H20 repo validation tests"
  for test_file in ${TEST_FILES}; do
    python3 "${test_file}"
  done
}

prepare_submodules
install_package

if [ "${RUN_TESTS}" = "1" ]; then
  validate_install
else
  log "Skipping tests because RUN_TESTS=0"
fi

log "Done"
