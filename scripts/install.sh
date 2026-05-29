#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT"

# Allow pip install in PEP 668 externally-managed environments (e.g. containers)
PYTHON_STDLIB=$(python -c "import sysconfig; print(sysconfig.get_path('stdlib'))")
if [ -f "${PYTHON_STDLIB}/EXTERNALLY-MANAGED" ]; then
  export PIP_BREAK_SYSTEM_PACKAGES=1
fi

python -m pip install -r requirements.txt

rm -rf build dist asym_gemm.egg-info
python -m pip install --no-build-isolation -e .

python -c "import asym_gemm; print('AsymGEMM', asym_gemm.__version__)"
