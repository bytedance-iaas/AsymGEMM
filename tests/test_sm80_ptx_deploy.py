# tests/test_sm80_ptx_deploy.py
"""
PTX forward-compatibility proof for the SM80 INT8 asym MoE kernel
(unified_kernel_sm80.md — "same artifact deploys on A100").

The production JIT compiles a per-device SASS cubin (csrc/jit/compiler.hpp:
`--gpu-architecture=sm_XX -cubin`), so H100/H200 runs of the test suite prove
the *source* is SM80-clean but not the deployment artifact. This test closes
that gap:

  1. nvcc -arch=compute_80 -ptx   -> pure virtual-arch PTX. This exact file is
     what an A100 driver would JIT to sm_80 SASS.
  2. cuModuleLoadData on the current device -> the driver JIT compiles the
     SAME PTX to native SASS (sm_90 on H200, sm_80 on A100). If any
     instruction were outside the compute_80 ISA, this step would fail.
  3. Launch through the raw driver API (mangled entry, packed
     SM80MoEInt8Params byte-struct) and compare against BOTH the float64
     reference and the production JIT-cubin path.

Passing here on an H200 + passing tests/test_arch_compile_gates.py means the
kernel is correct *as the artifact A100 will execute*, up to ptxas scheduling
differences — which cannot change INT8 MMA results (integer math is exact;
the FP32 epilogue has a fixed operation order in the PTX).

Run:  python -m pytest tests/test_sm80_ptx_deploy.py -v
"""
import ctypes
import ctypes.util
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import asym_gemm  # noqa: E402

NVCC = shutil.which("nvcc")
requires_env = pytest.mark.skipif(
    not torch.cuda.is_available() or NVCC is None,
    reason="needs CUDA device + nvcc")

INSTANTIATION_TU = """
#include <asym_gemm/impls/sm80_int8_asym_moe_gemm.cuh>
using namespace asym_gemm;
static void __instantiate_kernel() {
    auto p0 = reinterpret_cast<void*>(
        &sm80_int8_asym_moe_gemm_impl<128, 128, 128, 4>);
    (void)p0;
}
"""

BLOCK_M, BLOCK_N, BLOCK_K, NWARPS = 128, 128, 128, 4
SMEM_BYTES = (BLOCK_M + BLOCK_N) * BLOCK_K + (BLOCK_M + BLOCK_N) * 4


def build_compute80_ptx() -> str:
    """Compile the kernel TU to virtual-arch compute_80 PTX text."""
    include_dirs = [
        os.path.join(REPO_ROOT, "asym_gemm", "include"),
        os.path.join(REPO_ROOT, "third-party", "cutlass", "include"),
    ]
    with tempfile.TemporaryDirectory() as td:
        src, out = os.path.join(td, "k.cu"), os.path.join(td, "k.ptx")
        with open(src, "w") as f:
            f.write(INSTANTIATION_TU)
        cmd = [NVCC, "-std=c++17", "-arch=compute_80", "-ptx", "-O3",
               "--expt-relaxed-constexpr", "--expt-extended-lambda",
               "-o", out, src] + [f"-I{d}" for d in include_dirs]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, f"nvcc -arch=compute_80 -ptx failed:\n{r.stderr[-3000:]}"
        with open(out) as f:
            ptx = f.read()
    assert ".target sm_80" in ptx, "PTX is not targeted at virtual arch 80"
    return ptx


class _Cuda:
    """Minimal ctypes wrapper over the CUDA driver API."""

    def __init__(self):
        path = ctypes.util.find_library("cuda") or "libcuda.so.1"
        self.lib = ctypes.CDLL(path)

    def _check(self, res, what):
        if res != 0:
            s = ctypes.c_char_p()
            self.lib.cuGetErrorString(res, ctypes.byref(s))
            raise RuntimeError(f"{what}: CUresult={res} "
                               f"({(s.value or b'?').decode()})")

    def load_module(self, image: bytes):
        mod = ctypes.c_void_p()
        self._check(self.lib.cuModuleLoadData(ctypes.byref(mod),
                                              ctypes.c_char_p(image)),
                    "cuModuleLoadData (driver JIT of compute_80 PTX)")
        return mod

    def get_function(self, mod, name: str):
        fn = ctypes.c_void_p()
        self._check(self.lib.cuModuleGetFunction(ctypes.byref(fn), mod,
                                                 name.encode()),
                    f"cuModuleGetFunction({name})")
        return fn

    def launch(self, fn, grid, block, smem, stream, param_bytes: bytes):
        buf = ctypes.create_string_buffer(param_bytes, len(param_bytes))
        size = ctypes.c_size_t(len(param_bytes))
        CU_LAUNCH_PARAM_BUFFER_POINTER = ctypes.c_void_p(1)
        CU_LAUNCH_PARAM_BUFFER_SIZE = ctypes.c_void_p(2)
        CU_LAUNCH_PARAM_END = ctypes.c_void_p(0)
        extra = (ctypes.c_void_p * 5)(
            ctypes.cast(CU_LAUNCH_PARAM_BUFFER_POINTER, ctypes.c_void_p),
            ctypes.cast(ctypes.byref(buf), ctypes.c_void_p),
            ctypes.cast(CU_LAUNCH_PARAM_BUFFER_SIZE, ctypes.c_void_p),
            ctypes.cast(ctypes.byref(size), ctypes.c_void_p),
            CU_LAUNCH_PARAM_END)
        self._check(self.lib.cuLaunchKernel(
            fn, grid[0], grid[1], grid[2], block[0], block[1], block[2],
            smem, ctypes.c_void_p(stream), None, extra), "cuLaunchKernel")

    def unload(self, mod):
        self.lib.cuModuleUnload(mod)


def pack_params(a, b, d, experts, offsets, list_size, E, N, K, sfa, sfb, kb):
    """Byte-pack SM80MoEInt8Params exactly as the C ABI lays it out:
    5 pointers | 2 int32 | 2 int64 | 2 pointers | int32 + 4B tail padding."""
    return struct.pack(
        "<5QiiqqQQi4x",
        a.data_ptr(), b.data_ptr(), d.data_ptr(),
        experts.data_ptr(), offsets.data_ptr(),
        list_size, E, N, K,
        sfa.data_ptr(), sfb.data_ptr(), kb)


@requires_env
def test_compute80_ptx_driver_jit_parity():
    torch.manual_seed(11)
    dev = "cuda"

    ptx = build_compute80_ptx()
    entries = re.findall(r"\.visible \.entry (\w+)", ptx)
    kernels = [e for e in entries if "sm80_int8_asym_moe_gemm_impl" in e]
    assert len(kernels) == 1, f"expected 1 entry, got {entries}"
    print(f"\nPTX: .target sm_80, entry {kernels[0]}, {len(ptx) / 1e3:.0f} KB")

    # Problem: 3 experts, partial tiles, one gap segment, pinned-host B.
    S, N, K = 3, 256, 384
    kb = K // 128
    lens = [200, 70, 130]
    starts = [0, 200, 270]
    ends = [s + l for s, l in zip(starts, lens)]
    experts_l = [2, 0, 1]
    M = sum(lens)

    a = torch.randint(-127, 128, (M, K), dtype=torch.int8, device=dev)
    sfa = (torch.rand(M, kb, device=dev) + 0.1) * 0.01
    b = torch.randint(-127, 128, (S, N, K), dtype=torch.int8).pin_memory()
    sfb = ((torch.rand(S, N, kb) + 0.1) * 0.01).pin_memory()
    offsets = torch.tensor([v for se in zip(starts, ends) for v in se],
                           dtype=torch.int32, device=dev)
    experts = torch.tensor(experts_l + [-1], dtype=torch.int32, device=dev)

    # 1) Production path (per-device SASS cubin via the repo's JIT).
    d_prod = torch.zeros(M, N, dtype=torch.float32, device=dev)
    asym_gemm.m_grouped_int8_asym_gemm_sm80_contiguous(
        a, b, d_prod, offsets, experts, S + 1, sfa, sfb)
    torch.cuda.synchronize()

    # 2) Deployment artifact: the SAME compute_80 PTX an A100 would run,
    #    JIT-compiled by the local driver to this device's SASS.
    cuda = _Cuda()
    mod = cuda.load_module(ptx.encode() + b"\0")
    try:
        fn = cuda.get_function(mod, kernels[0])
        d_ptx = torch.zeros(M, N, dtype=torch.float32, device=dev)
        params = pack_params(a, b, d_ptx, experts, offsets,
                             S + 1, S, N, K, sfa, sfb, kb)
        stream = torch.cuda.current_stream().cuda_stream
        cuda.launch(fn, (N // BLOCK_N, S, 1), (NWARPS * 32, 1, 1),
                    SMEM_BYTES, stream, params)
        torch.cuda.synchronize()
    finally:
        cuda.unload(mod)

    # 3) Float64 reference.
    ref = torch.zeros(M, N, dtype=torch.float64, device=dev)
    bd, sd = b.cuda().to(torch.float64), sfb.cuda().to(torch.float64)
    for (s0, e0, e) in zip(starts, ends, experts_l):
        for k in range(kb):
            sl = slice(k * 128, (k + 1) * 128)
            ref[s0:e0] += ((a[s0:e0, sl].to(torch.float64) @ bd[e][:, sl].T)
                           * sfa[s0:e0, k, None].to(torch.float64)
                           * sd[e, None, :, k])

    rel_ptx = ((d_ptx.to(torch.float64) - ref).abs().max()
               / ref.abs().max()).item()
    bitwise = bool(torch.equal(d_ptx, d_prod))
    rel_cross = ((d_ptx - d_prod).abs().max()
                 / d_prod.abs().max().clamp(min=1e-30)).item()
    print(f"compute_80 PTX vs f64 reference: rel={rel_ptx:.3e}")
    print(f"compute_80 PTX vs production cubin: "
          f"{'BITWISE EQUAL' if bitwise else f'rel={rel_cross:.3e}'}")

    assert rel_ptx < 1e-4, f"PTX-deployed kernel wrong: rel={rel_ptx:.3e}"
    # ptxas scheduling may legally differ between driver JIT and offline
    # nvcc, but the FP32 epilogue op order is fixed in PTX — expect (near-)
    # bitwise agreement.
    assert rel_cross < 1e-6, f"PTX vs cubin diverged: rel={rel_cross:.3e}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
