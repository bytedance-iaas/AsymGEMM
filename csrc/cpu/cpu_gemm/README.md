# cpu_gemm

Standalone CPU GEMM (general matrix multiply) library, extracted and adapted from
the AMX/AVX kernels of
[ktransformers](https://github.com/kvcache-ai/ktransformers/tree/main)
(kvcache-ai, Apache-2.0). Targets the same workload — LLM forward passes with
diverse quantization — without dragging in the MoE driver, ggml, or pybind11.

Inside AsymGEMM it provides the **CPU bucket of the unified MoE runtime**
(`asym_gemm.unified_moe`, bound via `csrc/cpu/module.cpp` as `asym_gemm._cpu_C`),
but it builds and tests as a self-contained CMake project with a plain C ABI.

## Backends

| Backend             | Dtypes wired up                                   | State    |
|---------------------|---------------------------------------------------|----------|
| AMX-BF16            | `bf16 × bf16 → fp32`                              | landed   |
| AMX-INT8            | `int8 × int8 → fp32` (packed, prepacked-B, and stride-aware row-major) | landed   |
| AVX-512-VNNI        | `int8 × int8 → fp32` (row-major, `vpdpbusd`)      | landed   |
| AVX2 (FMA, no VNNI) | `bf16 × bf16 → fp32`                              | landed   |
| LA / AOCL           | —                                                 | planned  |

Backend selection happens once per process from a CPUID probe (plus a successful
`arch_prctl(ARCH_REQ_XCOMP_PERM)` call for AMX). For the INT8 row-major path the
preference order is AMX → AVX-512-VNNI; `ASYM_GEMM_FORCE_BACKEND={amx,avx512,none}`
overrides it for testing, and `cg_int8_rm_backend_name()` / `cg_int8_rm_backend_ok()`
report what was selected.

## Public surface

C ABI: `include/cpu_gemm/cpu_gemm.h`

- `cg_gemm()` — synchronous, parallel GEMM; `cg_gemm_st()` — single-thread slice
  for caller-owned parallelism.
- `cg_pack_b_int8_amx_size()` / `cg_pack_b_int8_amx()` — offline B pre-pack for
  the AMX INT8 path, so callers pay the packing cost once at model-load time
  (e.g. into pinned host memory) and reuse the buffer across calls
  (`dtype_b == CG_INT8_PACKED_AMX`).

C++ wrapper: `include/cpu_gemm/cpu_gemm.hpp` — RAII runtime handle, typed spans.

Runtime: `include/cpu_gemm/runtime.h` — portable `std::thread` work-stealing
pool plus the backend diagnostics above. A NUMA-aware pool can be slotted in
later behind a build option (`CPU_GEMM_WITH_NUMA=ON`).

## Building

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/examples/simple_bf16
```

From the repository root, `bash scripts/test_cpu_gemm.sh` configures, builds,
and runs the CTest suite in one step.

Requires a C++17 compiler with AVX2/FMA at minimum (Skylake / Zen 1 or newer);
AMX and AVX-512 kernels are compiled in by default (`CPU_GEMM_WITH_AMX`,
`CPU_GEMM_WITH_AVX512`, `CPU_GEMM_WITH_AVX2` CMake options) and selected at
runtime, so a single binary runs correctly on hosts without those ISAs.

## Layout

- `src/kernels/{amx,avx512,avx2}/` — ISA-specific kernels.
- `src/dispatch/` — dtype table and runtime INT8 row-major backend selection.
- `src/runtime/` — worker pool and scratch arena.
- `tests/`, `bench/`, `examples/` — CTest correctness suites (including
  AMX↔AVX-512 parity), microbenchmarks, and a minimal BF16 example.
- `analysis.md`, `avx_512.md` — design notes on the extraction and the
  AVX-512-VNNI kernel.
