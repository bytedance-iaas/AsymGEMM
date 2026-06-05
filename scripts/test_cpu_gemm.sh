#!/usr/bin/env bash
# Build and run the vendored cpu_gemm test suite. Maintainer use — confirms
# the third-party/cpu_gemm snapshot is healthy after sync_cpu_gemm.sh.
#
# Independent of the asym_gemm pip install: invokes cpu_gemm's own CMake.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC="$ROOT/third-party/cpu_gemm"
BLD="$ROOT/build/cpu_gemm_tests"

if [[ ! -d "$SRC" ]]; then
  echo "[FATAL] Vendored cpu_gemm not found at: $SRC" >&2
  exit 1
fi

cmake -S "$SRC" -B "$BLD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BLD" -j
ctest --test-dir "$BLD" --output-on-failure
