#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT"

# Vendored cutlass/fmt are git submodules; a fresh clone has them empty.
if [ -d .git ] && command -v git >/dev/null 2>&1; then
  git submodule update --init --recursive
fi

# Allow pip install in PEP 668 externally-managed environments (e.g. containers)
PYTHON_STDLIB=$(python -c "import sysconfig; print(sysconfig.get_path('stdlib'))")
if [ -f "${PYTHON_STDLIB}/EXTERNALLY-MANAGED" ]; then
  export PIP_BREAK_SYSTEM_PACKAGES=1
fi

python -m pip install -r requirements.txt

# Surface a clearer error than letting CMake fail mid-build of cpu_gemm.
if ! command -v cmake >/dev/null 2>&1; then
  echo "[FATAL] cmake (>= 3.18) is required to build the vendored cpu_gemm." >&2
  echo "        Install with e.g. \`apt install cmake\` or \`pip install cmake\`." >&2
  exit 1
fi

rm -rf build dist asym_gemm.egg-info
python -m pip install --no-build-isolation -e .

python - <<'PY'
import asym_gemm
print(f"AsymGEMM {asym_gemm.__version__}")

# GPU extension
print(f"  GPU extension (_C): {'YES' if hasattr(asym_gemm, '_C') else 'NOT BUILT'}")

# CPU extension + unified MoE
if asym_gemm.unified_moe is not None:
    caps = asym_gemm.unified_moe._C.caps()
    amx = "YES (runtime)" if caps["has_amx_int8"] else "no (runtime — host lacks AMX-INT8)"
    print(f"  CPU extension (_cpu_C): YES")
    print(f"  AMX-INT8 path:           {amx}")
else:
    print(f"  CPU extension (_cpu_C): NOT BUILT (unified_moe disabled)")
PY
