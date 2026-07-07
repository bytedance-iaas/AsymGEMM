#!/usr/bin/env bash
# Build and run the cpu_gemm test suite (csrc_cpu/cpu_gemm).
#
# Independent of the asym_gemm pip install: invokes cpu_gemm's own CMake.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC="$ROOT/csrc_cpu/cpu_gemm"
BLD="$ROOT/build/cpu_gemm_tests"

if [[ ! -d "$SRC" ]]; then
  echo "[FATAL] cpu_gemm sources not found at: $SRC" >&2
  exit 1
fi

cmake -S "$SRC" -B "$BLD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BLD" -j
ctest --test-dir "$BLD" --output-on-failure
