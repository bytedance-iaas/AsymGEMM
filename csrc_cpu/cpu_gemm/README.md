# cpu_gemm

Standalone CPU GEMM (general matrix multiply) library extracted from
[ktransformers/kt-kernel](https://github.com/kvcache-ai/ktransformers).
Targets the same workload — LLM forward passes with diverse quantization —
without dragging in the MoE driver, ggml, or pybind11.

## Status

**v0.1 work in progress.** Vertical slice through one backend (AVX2 BF16) is
the first deliverable; subsequent backends (AMX, AVX‑512, AVX‑VNNI, LA, AOCL)
are parallel add-on work tracked in `cpu_gemm.md` in the parent directory.

| Backend             | Dtypes wired up so far  | State    |
|---------------------|-------------------------|----------|
| AVX2 (FMA, no VNNI) | `bf16 × bf16 → fp32`    | landed   |
| AMX-BF16            | `bf16 × bf16 → fp32`    | landed   |
| AVX-512             | -                       | planned  |
| AVX-VNNI            | -                       | planned  |
| LA / AOCL           | -                       | planned  |

The dispatcher prefers AMX → AVX-512 → AVX2 at runtime based on a CPUID
probe and a successful `arch_prctl(ARCH_REQ_XCOMP_PERM)` call.

## Public surface

C ABI: `include/cpu_gemm/cpu_gemm.h` — one synchronous `cg_gemm()` plus a
single-thread entry `cg_gemm_st()` for caller-owned parallelism.

C++ wrapper: `include/cpu_gemm/cpu_gemm.hpp` — RAII runtime handle, typed
spans.

Runtime: `include/cpu_gemm/runtime.h` — portable `std::thread` work‑stealing
pool. The NUMA-aware version from ktransformers can be slotted in later
behind a build option (`CPU_GEMM_WITH_NUMA=ON`).

## Building

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/examples/simple_bf16
```

Requires a C++17 compiler with AVX2/FMA at minimum (Skylake / Zen 1 or
newer).

## Layout

See `cpu_gemm.md` (in the parent directory) for the rationale and the full
extraction plan.
