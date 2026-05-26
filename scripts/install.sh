#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT"

python -m pip install -r requirements.txt

rm -rf build dist asym_gemm.egg-info
python -m pip install --no-build-isolation -e .

python -c "import asym_gemm; print('AsymGEMM', asym_gemm.__version__)"
