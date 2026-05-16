#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

rm -rf ~/.asym_gemm/cache
python3 tests/m_grouped/test_h20_fp8.py

echo "Done"
