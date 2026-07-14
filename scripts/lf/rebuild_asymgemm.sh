#!/bin/bash
# =============================================================================
# Canonical rebuild of AsymGEMM's `_C` CUDA extension.
#
# MUST run INSIDE the enroot container (e.g. `asym40_enroot_run`), NEVER on the host.
#
# WHY THIS SCRIPT EXISTS
#   `_C` links against libtorch, so it MUST be compiled with the SAME python/torch
#   that will RUN it. The training venvs (.venv, .venv-fa4) ship torch 2.12.0; the
#   container's system python ships a DIFFERENT torch. Building `_C` with the wrong
#   interpreter produces a .so that fails to load at runtime with:
#       ImportError: _C...so: undefined symbol: _ZN3c104cuda29c10_cuda_check_impl...
#   after which every asym GEMM call degrades to 'missing_bf16_asym_binding' and the
#   training run dies on the first forward. Always rebuild through THIS script.
#
# WHAT IT DOES  (mirrors the bootstrap step: `pip install --no-build-isolation -e .`)
#   * cleans build/ + stale _C*.so so header-only edits (e.g. csrc/jit/compiler.hpp)
#     force a full recompile — setuptools does NOT track header dependencies.
#   * builds the editable extension with --no-build-isolation => links against the
#     venv's own torch (correct ABI), --no-deps => leaves pinned deps untouched.
#   * verifies the fresh .so actually LOADS and BINDS under every present venv
#     (.venv and .venv-fa4 share the same py3.12 + torch 2.12 ABI, so one .so serves
#     both).
#
# USAGE
#   asym40_enroot_run                               # then, inside the container:
#   bash third_party/AsymGEMM/scripts/lf/rebuild_asymgemm.sh
#   # or non-interactively via a wrapper: bash .../scripts/lf/rebuild_asymgemm.sh
#   BUILD_VENV=.venv-fa4 bash .../scripts/lf/rebuild_asymgemm.sh   # build via fa4 env
# =============================================================================
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)   # repo root: .../third_party/AsymGEMM
cd "$ROOT"

BUILD_VENV=${BUILD_VENV:-.venv}
BUILD_PY="$ROOT/$BUILD_VENV/bin/python"
if [[ ! -x "$BUILD_PY" ]]; then
  echo "[rebuild] ERROR: $BUILD_PY not found." >&2
  echo "[rebuild]        Run INSIDE the container and bootstrap the venv first" >&2
  echo "[rebuild]        (scripts/lf/bootstrap_lf_venv.sh / bootstrap_lf_venv_fa4.sh)." >&2
  exit 2
fi

TV=$("$BUILD_PY" -c 'import torch; print(torch.__version__)')
echo "[rebuild] building _C against $BUILD_VENV (torch $TV) --no-build-isolation --no-deps -e ."

echo "[rebuild] cleaning build/ and stale asym_gemm/_C*.so (force full recompile)"
rm -rf "$ROOT/build" "$ROOT"/asym_gemm/_C*.so

"$BUILD_PY" -m pip install --no-build-isolation --no-deps -e "$ROOT"

echo "[rebuild] verifying the fresh .so loads + binds under each present venv"
rc=0
for v in .venv .venv-fa4; do
  PY="$ROOT/$v/bin/python"
  [[ -x "$PY" ]] || continue
  if PYTHONPATH="$ROOT" "$PY" - <<'PY'
import sys
try:
    import asym_gemm
    from asym_gemm import _C  # triggers the ABI symbol resolution
except Exception as e:
    print("    import FAILED:", type(e).__name__, str(e)[:180]); sys.exit(1)
ok = hasattr(asym_gemm, "m_grouped_bf16_asym_gemm_nt_contiguous")
print("    bf16 asym binding present:", ok)
sys.exit(0 if ok else 1)
PY
  then echo "[rebuild]   OK under $v"; else echo "[rebuild]   FAILED under $v"; rc=1; fi
done

if [[ $rc -eq 0 ]]; then
  echo "[rebuild] SUCCESS — _C rebuilt and verified."
else
  echo "[rebuild] FAILURE — see errors above." >&2
  exit 1
fi
