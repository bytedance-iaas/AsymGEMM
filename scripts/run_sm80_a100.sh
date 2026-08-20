#!/usr/bin/env bash
# One-command SM80/A100 validation + benchmark runner (unified_kernel_sm80.md).
#
# Usage (on the A100 server, after cloning the repo):
#   bash scripts/run_sm80_a100.sh
#
# Also runnable on the H100/H200 dev box to record the comparison baseline —
# the SM80 kernels' arch guard is >= 800, so the identical code path runs
# on any Ampere-or-newer data-center GPU.
#
# Steps: env report -> build -> clear JIT cache -> sm_80/sm_89 compile gates
#        -> INT8 kernel parity suite -> unified Layer tests (self-skips
#        without an AMX/VNNI CPU) -> performance benchmark (+ baseline diff
#        when bench_results/sm80_int8_h200.json is present).
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "==================================================================="
echo "[1/6] Environment"
echo "==================================================================="
nvidia-smi --query-gpu=name,compute_cap,memory.total,pcie.link.gen.current,pcie.link.width.current --format=csv || true
"${PYTHON_BIN}" -c "import torch; print('torch', torch.__version__, '| CUDA', torch.version.cuda, '| device cc %d.%d' % torch.cuda.get_device_capability(0))"
nvcc --version | tail -1

SM=$("${PYTHON_BIN}" -c "import torch; print('%d%d' % torch.cuda.get_device_capability(0))")
if [[ "$SM" != "80" ]]; then
  echo "[NOTE] Device is sm_${SM}, not sm_80 (A100). The suite still exercises"
  echo "       the exact SM80 code path; results serve as the comparison baseline."
fi

echo
echo "==================================================================="
echo "[2/6] Build host extension"
echo "==================================================================="
"${PYTHON_BIN}" setup.py build_ext --inplace

echo
echo "==================================================================="
echo "[3/6] Clear JIT cache (standing project rule before gate runs)"
echo "==================================================================="
rm -rf "${DG_JIT_CACHE_DIR:-$HOME/.asym_gemm}/cache" "${DG_JIT_CACHE_DIR:-$HOME/.asym_gemm}/tmp"
echo "cleared ${DG_JIT_CACHE_DIR:-$HOME/.asym_gemm}"

echo
echo "==================================================================="
echo "[4/6] Architecture compile gates (nvcc -arch=sm_80 / sm_89)"
echo "==================================================================="
"${PYTHON_BIN}" -m pytest tests/test_arch_compile_gates.py -q

echo
echo "==================================================================="
echo "[5/6] Correctness: SM80 INT8 parity + unified Layer"
echo "==================================================================="
"${PYTHON_BIN}" -m pytest tests/test_sm80_int8_asym.py -q
# Deployment-artifact proof: compute_80 PTX loaded via the driver JIT must
# match the production cubin bitwise (on A100 this JITs to native sm_80 SASS).
"${PYTHON_BIN}" -m pytest tests/test_sm80_ptx_deploy.py -q
# Self-skips (exit 0) when the host CPU lacks AMX; on such hosts the CPU
# bucket is served by AVX512-VNNI or disabled — see caps() in _cpu_C.
"${PYTHON_BIN}" tests/test_unified_moe.py

echo
echo "==================================================================="
echo "[6/6] Performance benchmark"
echo "==================================================================="
HOSTTAG=$("${PYTHON_BIN}" -c "import torch,re; print(re.sub(r'[^a-z0-9]+','_',torch.cuda.get_device_name(0).lower()).strip('_'))")
BASELINE_ARG=()
if [[ "$SM" == "80" && -f bench_results/sm80_int8_h200.json ]]; then
  BASELINE_ARG=(--baseline bench_results/sm80_int8_h200.json)
fi
"${PYTHON_BIN}" tests/bench_sm80_int8.py \
  --save "bench_results/sm80_int8_${HOSTTAG}.json" "${BASELINE_ARG[@]}"

echo
echo "All SM80 gates green. Results: bench_results/sm80_int8_${HOSTTAG}.json"
