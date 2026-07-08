# cpu_gemm — Implementation Analysis (current state of this tree)

This document reviews the code that actually exists in `/workspace/cpu_gemm/`,
as opposed to the upstream survey (`../cpu_gemm_analysis.md`, which analyzes
ktransformers) and the roadmap (`../cpu_gemm.md`). It records what is wired up,
what is tested, the bugs found during review, the fixes applied, and what is
still missing relative to the plan.

Scope of this pass: read every source/header/test/bench file, build with the
project's `-Wall -Wextra` flags, run the test suite, prove a memory-safety bug
with a guard-page harness, fix it, and add coverage for the one fully-written
but completely untested kernel.

---

## 1. What the library is

A standalone C/C++ GEMM library extracted from ktransformers/kt-kernel, exposing
a single synchronous call

```
C = alpha * op(A) * op(B) + beta * C
```

behind a CBLAS-shaped C ABI (`include/cpu_gemm/cpu_gemm.h`). The intended scope
(per `../cpu_gemm.md`) is the general-purpose GEMM machinery — kernels,
threading, scratch — with the MoE-specific plumbing left in the downstream
caller.

Layering, bottom to top:

| Layer | Files | Role |
|---|---|---|
| Public ABI | `include/cpu_gemm/{cpu_gemm.h, cpu_gemm.hpp, types.h, runtime.h}` | C ABI + thin RAII C++ wrapper + dtype/caps enums |
| Dispatch | `src/dispatch/{gemm.cpp, dtype.cpp}` | Descriptor validation, backend selection, pack→run→unpack orchestration |
| Runtime | `src/runtime/{worker_pool, scratch_arena, runtime}.*` | `std::thread` work-stealing pool, growable aligned scratch, opaque `cg_runtime` |
| Kernels | `src/kernels/avx2/`, `src/kernels/amx/`, `src/kernels/bf16_compat.h` | The actual SIMD/AMX math + packed-buffer layouts |

The kernels are single-threaded and take `(ith, nth)` slicing the **N**
dimension; parallelism is the runtime's job. This matches the upstream design.

---

## 2. What is implemented and reachable

### 2.1 Backend / dtype matrix actually wired

| Path | dtype_a | dtype_b | dtype_c | Entry | Reachable via |
|---|---|---|---|---|---|
| AVX2 | BF16 | BF16 (`CG_TRANS`) | F32 | `kernels::avx2::gemm_bf16_bf16_f32` | `cg_gemm` + `cg_gemm_st` |
| AMX-BF16 | BF16 | BF16 (`CG_TRANS`) | F32 | `kernels::amx::bf16_{pack,run,unpack}` | `cg_gemm` only (needs scratch) |
| AMX-INT8 | BF16 | INT8 (`CG_TRANS`) + `b_scales` | F32 | `kernels::amx::int8_*` | `cg_gemm` only |

Everything else in the `cg_dtype_t` enum (F16, the FP8 family, MXFP4, the INT4
family) is **declared but not implemented** — `dtype.cpp` knows their sizes and
names, but the dispatcher returns `CG_E_UNSUPPORTED`.

### 2.2 Dispatch logic (`src/dispatch/gemm.cpp`)

- `cg_gemm_st`: AVX2 BF16 only. Pure function; caller owns parallelism.
- `cg_gemm`: priority **AMX-INT8 → AMX-BF16 → AVX2 fan-out**.
  - AMX eligibility (`amx_bf16_eligible` / `amx_int8_eligible`) requires:
    runtime `amx_available()`, the exact dtype/trans tuple, `k % K_STEP == 0`
    (32 for BF16, 64 for INT8), and `lda == k && ldb == k` (the packer is not
    stride-aware yet). When any check fails it falls through to the AVX2
    fan-out, which itself only handles BF16×BF16.
- The AMX paths own the scratch arena and run three sequential `parallel_for`
  phases (pack → run → unpack), partitioning N into `N_BLOCK` chunks and
  capping `nth` at `ceil(n / N_BLOCK)` so surplus threads don't idle.

### 2.3 Runtime

- `WorkerPool` (`worker_pool.cpp`): condition-variable wakeup, atomic
  work-stealing counter, calling thread participates as worker 0. Correct and
  simple; no NUMA (deferred behind a future `CPU_GEMM_WITH_NUMA`).
- `ScratchArena`: mutex-guarded, geometric grow, 64-byte aligned via
  `posix_memalign`, never shrinks. One arena per runtime, reused across calls.
- `cg_query_caps`: CPUID probe for AVX2/FMA/AVX512F/AVX512-BF16/AVX-VNNI/
  AMX-BF16/AMX-INT8.

### 2.4 AMX kernels

Faithful ports of the ktransformers `GemmKernel224*` family: 2×2 tiles of
16×16, `_tile_dpbf16ps` (BF16) / `_tile_dpbssd` (INT8), the 16×16 32-bit VNNI
transpose during B packing, per-thread `enable_amx()` permission grant guarded
by `__attribute__((noinline))` + an asm memory barrier to stop the optimizer
hoisting `ldtilecfg` above the `arch_prctl`. INT8 quantizes A row-wise
(`amax/127`) on the fly and rescales the int32 accumulator by
`a_scale[i] * b_scale[j]` during unpack.

---

## 3. Build and test status

- Builds clean (`gcc 13.3`, C++17, Release) with no `-Wall -Wextra` warnings.
- `ctest`: **4/4 pass** on this host (AVX2-BF16, runtime, AMX-BF16, AMX-INT8).
- Host probed: AVX2/FMA/AVX512/AVX-VNNI/AMX-BF16/AMX-INT8 all present, 192
  hardware threads.

Performance snapshot (`bench_amx_vs_avx2`, GFLOPS, higher is better):

| Shape | AVX2 (32T) | AMX (192T) | speedup |
|---|---|---|---|
| decode    m=1   n=4096  k=4096  | 26  | 18   | 0.7× |
| decode    m=1   n=14336 k=4096  | 72  | 46   | 0.6× |
| prefill   m=8   n=4096  k=4096  | 179 | 137  | 0.8× |
| balanced  m=32  n=4096  k=4096  | 419 | 561  | 1.3× |
| ffn-up    m=64  n=14336 k=4096  | 648 | 2675 | 4.1× |
| wide      m=256 n=4096  k=4096  | 673 | 2852 | 4.2× |

Takeaway: AMX dominates for batched/prefill shapes but is **slower than AVX2
for decode (M=1..8)**, where the pack overhead and AMX tile under-utilization
outweigh the raw throughput. See §5.

---

## 4. Bugs found and fixed in this pass

### 4.1 [FIXED] Out-of-bounds read when N is not a multiple of N_STEP

**Severity: high (memory safety).** When the AMX path was selected and `n` was
not a multiple of `N_STEP` (32), the B packer padded `n` up to `n_pad` and then
read source rows `[n, n_pad)` — past the end of the caller's weight buffer.
Affected `BufferBBF16::from_mat` (bf16) and `BufferBInt8::from_bf16` /
`from_int8` (int8); the int8 `from_int8` also over-read the caller's `b_scales`.

This was latent because the existing tests allocate B as `std::vector` with
heap slack after it, so the over-read landed in valid memory and the garbage
columns were discarded by the unpack step. Proven real with a guard-page
(`PROT_NONE`) harness: `cg_gemm` with `n=71` and B abutting an unmapped page
**SIGSEGV'd** before the fix and returns cleanly after, with bit-identical
output for the valid columns.

Fix: thread the real `n` into the packers and zero-fill padded tile rows in the
packed buffer instead of reading the source. (`src/kernels/amx/bf16_buffers.h`,
`int8_buffers.h`, callers in `bf16_gemm.cpp` / `int8_gemm.cpp`.)

A subtlety worth recording: the first attempt zeroed padded scales with a
per-element ternary `(row < n_real) ? scales_block[i] : 0.0f`. At `-O2` GCC
auto-vectorized that into an **unconditional** vector load of `scales_block[i]`
for the whole vector width (then a blend), which re-introduced the same
over-read and faulted. The committed fix uses an explicit loop bound
(`real_in_block`) so no speculative load past `n_real` is emitted. The
data-copy paths use a plain `if/else` branch (the compiler will not speculate a
faulting load across it), so only the scale copies needed restructuring.

### 4.2 [FIXED-by-coverage] AMX INT8 path had zero tests

`int8_gemm.cpp` + `int8_buffers.h` (~350 LoC) were fully implemented and
reachable through `cg_gemm`, but **no test exercised them** — dead-on-arrival
code. Added `tests/test_amx_int8.cpp` with an FP32 reference that mirrors the
kernel's row-wise A quantization, covering aligned/unaligned N, single- and
multi-threaded, and an alpha/beta sweep. All cases match the reference
**exactly** (abs = rel = 0), which also confirms the §4.1 zero-fill does not
perturb valid output.

---

## 5. Observations and recommended next steps (not changed in this pass)

1. **Dispatch is not M-aware.** `cg_gemm` always prefers AMX when eligible, but
   the bench shows AMX losing 20–40% to AVX2 at M=1..8. Upstream uses an
   AVX-512 path "for very small M". Recommendation: route small-M (decode)
   shapes to the non-packing AVX2 fan-out, or add the AVX-512 fallback.

2. **`int8_pack_b_bf16` is unreachable.** The "B is BF16, quantize per output
   channel to int8" path exists (`int8_gemm.cpp:160`) but the dispatcher only
   wires `int8_pack_b_int8` (caller supplies int8 + scales). Either expose it
   (e.g. a dtype_b distinction) or drop it.

3. **Stride-locked AMX.** AMX eligibility requires `lda == k && ldb == k`. Any
   caller with a real leading dimension silently falls back to AVX2 (BF16) or
   `CG_E_UNSUPPORTED` (int8). A stride-aware packer would remove the surprise.

4. **Padded A rows are read uninitialized (benign).** The A packers fill only
   real rows `[0, m)`; the kernel loads full `M_STEP` tiles including padded
   rows from reused scratch. Those feed only discarded output rows, so results
   are unaffected — but it is an uninitialized in-bounds read. Zeroing the A
   pad would remove the UB if a sanitizer lane is ever added.

5. **`cg_gemm_st` covers only AVX2 BF16.** The single-thread entry can't reach
   the AMX paths (they need scratch). Documented, but worth a note in the
   header for callers expecting full coverage.

6. **No AVX-512 / AVX-VNNI / FP8 / INT4 backends yet.** These are planned work
   (`../cpu_gemm.md` steps 3–5); the enum and CMake placeholders exist but the
   kernels do not. The dispatcher's fall-through structure already accommodates
   them.

7. **Tooling gaps vs the plan.** No `cg_pack_b` offline-packing C ABI (the AMX
   paths repack B every call), no `ggml_compat.h`, no CI matrix. Re-packing B
   on every call is the largest latent perf cost for repeated-weight workloads.

---

## 6. File-level index

```
include/cpu_gemm/         public surface (C ABI, C++ RAII, dtypes, caps)
src/dispatch/gemm.cpp     descriptor validation + backend selection + phases
src/dispatch/dtype.cpp    cg_dtype_bits / cg_dtype_name
src/runtime/              worker pool, scratch arena, runtime handle, CPUID
src/kernels/avx2/         BF16xBF16->F32 (FMA, scalar tail)
src/kernels/amx/          tile config + BF16 and INT8 kernels + packed buffers
src/kernels/bf16_compat.h standalone BF16<->FP32 (ggml RNE semantics)
tests/                    avx2_bf16, amx_bf16, amx_int8 (new), runtime, probe_amx
bench/                    bf16 throughput, amx-vs-avx2 head-to-head
examples/simple_bf16.cpp  minimal standalone usage
```
