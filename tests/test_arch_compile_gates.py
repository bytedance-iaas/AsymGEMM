"""Compile-only architecture gates for the SM80-family MoE kernels.

Compiles each device header's kernel instantiations with nvcc for the *minimum*
architecture the header claims to support, without launching anything. This is
the hard guarantee that no SM89/SM90-only instruction leaks into a header that
must run on A100 (SM80) — ptxas hard-errors on instructions the target arch
lacks. Runs on any box with nvcc; no GPU required.

Gates:
  sm80_moe_gemm.cuh           → -arch=sm_80  (A100)
  sm80_int8_asym_moe_gemm.cuh → -arch=sm_80  (A100)   [Phase 1]
  sm89_fp8_moe_gemm.cuh       → -arch=sm_89  (Ada)
"""

import os
import shutil
import subprocess
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INCLUDE_DIRS = [
    os.path.join(REPO_ROOT, "asym_gemm", "include"),
    os.path.join(REPO_ROOT, "third-party", "cutlass", "include"),
]

NVCC = shutil.which("nvcc")


def _compile_only(source: str, arch: str) -> None:
    """Compile a CUDA TU to cubin (compile-only) for the given arch; assert success."""
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "gate.cu")
        out = os.path.join(td, "gate.cubin")
        with open(src, "w") as f:
            f.write(source)
        cmd = [
            NVCC, "-std=c++17", f"-arch={arch}", "-cubin", "-o", out, src,
            "--expt-relaxed-constexpr", "--expt-extended-lambda",
        ] + [f"-I{d}" for d in INCLUDE_DIRS]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, (
            f"nvcc -arch={arch} failed:\n{r.stderr[-4000:]}"
        )


@pytest.mark.skipif(NVCC is None, reason="nvcc not on PATH")
def test_sm80_moe_gemm_compiles_for_sm80():
    _compile_only(
        """
#include <asym_gemm/impls/sm80_moe_gemm.cuh>
using namespace asym_gemm;
static void __instantiate_kernel() {
    auto p0 = reinterpret_cast<void*>(
        &sm80_moe_gemm_impl<128, 128, 64, 4, cutlass::bfloat16_t>);
    auto p1 = reinterpret_cast<void*>(
        &sm80_moe_gemm_impl<128, 64, 128, 4, cutlass::half_t>);
    (void)p0; (void)p1;
}
""",
        "sm_80",
    )


@pytest.mark.skipif(NVCC is None, reason="nvcc not on PATH")
def test_sm89_fp8_moe_gemm_compiles_for_sm89():
    _compile_only(
        """
#include <asym_gemm/impls/sm89_fp8_moe_gemm.cuh>
using namespace asym_gemm;
static void __instantiate_kernel() {
    auto p0 = reinterpret_cast<void*>(
        &sm89_moe_fp8_gemm_impl<128, 128, 128, 4, false>);
    auto p1 = reinterpret_cast<void*>(
        &sm89_moe_fp8_gemm_impl<128, 128, 128, 4, true>);
    auto p2 = reinterpret_cast<void*>(
        &sm89_moe_fp8_gemm_masked_impl<128, 64, 128, 4, false>);
    auto p3 = reinterpret_cast<void*>(
        &sm89_moe_fp8_gemm_masked_impl<128, 64, 128, 4, true>);
    (void)p0; (void)p1; (void)p2; (void)p3;
}
""",
        "sm_89",
    )


@pytest.mark.skipif(NVCC is None, reason="nvcc not on PATH")
def test_sm80_int8_asym_moe_gemm_compiles_for_sm80():
    header = os.path.join(
        REPO_ROOT, "asym_gemm", "include", "asym_gemm", "impls",
        "sm80_int8_asym_moe_gemm.cuh")
    if not os.path.exists(header):
        pytest.skip("sm80_int8_asym_moe_gemm.cuh not present yet (Phase 1)")
    _compile_only(
        """
#include <asym_gemm/impls/sm80_int8_asym_moe_gemm.cuh>
using namespace asym_gemm;
static void __instantiate_kernel() {
    auto p0 = reinterpret_cast<void*>(
        &sm80_int8_asym_moe_gemm_impl<128, 128, 128, 4>);
    auto p1 = reinterpret_cast<void*>(
        &sm80_int8_asym_moe_gemm_masked_impl<128, 64, 128, 4>);
    (void)p0; (void)p1;
}
""",
        "sm_80",
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
