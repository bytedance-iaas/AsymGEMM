#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

rm -rf ~/.asym_gemm/cache
pytest -q tests/m_grouped/test_h20_bf16.py tests/m_grouped/test_h20_fp8.py -s
PYTHONPATH=. pytest -q tests/training -s

if [[ "${RUN_MEMORY:-1}" != "0" || "${RUN_TIMING:-1}" != "0" ]]; then
  python3 scripts/benchmark_h20.py
fi

echo "Done"
