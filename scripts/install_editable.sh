#!/bin/bash
# Editable development install for AsymGEMM.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0a}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "==> Installing asym_gemm editable"
python3 -m pip uninstall -y asym_gemm || true
python3 -m pip install -e . --no-build-isolation -v

echo
echo "==> Verifying editable install"
python3 - <<'PY'
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
