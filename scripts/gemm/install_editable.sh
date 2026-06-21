#!/bin/bash
# Editable development install for AsymGEMM.

set -euo pipefail

# =============================================================================
# User Parameters
# =============================================================================
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0a}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DG_JIT_WITH_LINEINFO="${DG_JIT_WITH_LINEINFO:-1}"
export DG_JIT_CLEAR_CACHE="${DG_JIT_CLEAR_CACHE:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# =============================================================================
# Derived Parameters
# =============================================================================
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# =============================================================================
# Main Logic
# =============================================================================
cd "$ROOT_DIR"

echo "==> Installing asym_gemm editable"
echo "    DG_JIT_WITH_LINEINFO=${DG_JIT_WITH_LINEINFO}"
echo "    DG_JIT_CLEAR_CACHE=${DG_JIT_CLEAR_CACHE}"
if [[ "${DG_JIT_CLEAR_CACHE}" == "1" ]]; then
  echo "==> Clearing AsymGEMM JIT cache: ${HOME}/.asym_gemm/cache"
  rm -rf "${HOME}/.asym_gemm/cache"
fi
rm -rf build *.egg-info
"${PYTHON_BIN}" -m pip uninstall -y asym_gemm || true
"${PYTHON_BIN}" -m pip install -e . --no-build-isolation -v

echo
echo "==> Verifying editable install"
"${PYTHON_BIN}" - <<'PY'
import asym_gemm

print("asym_gemm:", asym_gemm.__file__)
print("_C:", asym_gemm._C.__file__)
print("version:", asym_gemm.__version__)
required = [
    "m_grouped_fp8_asym_gemm_nt_contiguous",
    "m_grouped_fp8_asym_gemm_nt_masked",
]
missing = [name for name in required if not hasattr(asym_gemm, name)]
if missing:
    raise RuntimeError(f"missing exported kernels: {missing}")
print("editable install check: PASS")
PY

echo
echo "Done"
