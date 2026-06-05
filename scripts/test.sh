#!/usr/bin/env bash
# Arch-aware AsymGEMM test runner.
# Detects the current GPU's compute capability and runs only the test files
# that apply to that arch. Prints a pass/fail summary at the end. See
# docs/e2e_test.md for the full design rationale and arch→tests table.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TEST_DIR="$ROOT_DIR/tests"
cd "$ROOT_DIR"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

export PYTHONPATH="$ROOT_DIR:$TEST_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Detect SM via torch — already a hard dependency of asym_gemm, so this avoids
# parsing nvidia-smi text and matches what asym_gemm.testing.get_arch_major uses.
SM=$(python - <<'PY'
try:
    import torch
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        print(f"{major}{minor}")
    else:
        print("none")
except Exception:
    print("none")
PY
)

case "$SM" in
  89)  TESTS=(tests/test_sm89_moe.py) ;;
  90)  TESTS=(tests/test_bf16_asym_gemm.py tests/test_fp8_asym_gemm.py) ;;
  100) TESTS=(tests/test_bf16_asym_gemm.py tests/test_fp8_asym_gemm.py tests/test_fp4_asym_gemm.py) ;;
  none)
    echo "[FATAL] No CUDA GPU detected. AsymGEMM tests require a CUDA device." >&2
    exit 1
    ;;
  *)
    echo "[WARN] Compute capability sm_${SM} is not covered by any AsymGEMM test; nothing to run." >&2
    exit 0
    ;;
esac

# Unified MoE parity tests — always run; internal-skip when AMX is absent.
TESTS+=(tests/test_unified_moe.py)

echo "Detected sm_${SM} — running ${#TESTS[@]} test file(s):"
printf "  - %s\n" "${TESTS[@]}"

declare -a PASSED=() FAILED=()
for f in "${TESTS[@]}"; do
  echo
  echo "=========================================================="
  echo "Running: $f"
  echo "=========================================================="
  if python "$f"; then
    PASSED+=("$f")
  else
    FAILED+=("$f")
  fi
done

echo
echo "=========================================================="
echo "Summary (sm_${SM}):"
echo "=========================================================="
for t in "${PASSED[@]}"; do echo "  PASS  $t"; done
for t in "${FAILED[@]}"; do echo "  FAIL  $t"; done

[[ ${#FAILED[@]} -eq 0 ]] && exit 0 || exit 1
