# INT8 SM90 Asymmetric GEMM — Implementation Plan

`sm90_int8_asym_gemm_1d1d.cuh`

---

## STATUS: IMPLEMENTED & VERIFIED (H200 / SM90)

Everything below was implemented and all verification stages (§7 A–F) pass on an
NVIDIA H200 with `calc_diff == 0.0` (kernel reproduces the exact integer math; the
only deviation is FP32 rounding, ~1e-14):

```
smoke (k=128, STORE only):            diff=0.00000
smoke (k=512, REDUCE_ADD):            diff=0.00000
masked  G=1 n=256 k=512:              max_diff=0.00000
masked  G=4 n=256 k=512:              max_diff=0.00000
masked  G=8 n=128 k=256:              max_diff=0.00000
contiguous G=2 m=512 n=256 k=512:     diff=0.00000
contiguous G=4 m=1024 n=128 k=256:    diff=0.00000
Stage F (saturating ±127, K=1024, G=4): max_diff=2.5e-14, deterministic
```

### Deviations from the original plan (and why)

1. **`BLOCK_M = 64`, not 128.** The FP8 progenitor's epilogue writes the register
   accumulator to `smem_cd` with row index `wg_local_warp_idx*16 + …` and **no
   `math_wg_idx*WGMMA::M` offset**. With `BLOCK_M=128` (2 math warp-groups) both
   warp-groups would write rows 0–63 and rows 64–127 would be lost. `BLOCK_M=64`
   uses a single math warp-group (128 math threads) and is correct as-is.
2. **Two FP8-progenitor bugs fixed in the int8 kernel** (both only bite the FP32
   output path, which the FP8 SM90 asym kernel was never wired to exercise):
   * **CD smem size** used `kSwizzleCDMode` (= 0 for FP32) → 0-byte tile. Replaced
     with `CD_ROW_BYTES = kSwizzleCDMode==0 ? BLOCK_N*sizeof(float) : kSwizzleCDMode`.
   * **CD store N coordinate** used the group-global `scheduler.n_idx`; groups are
     folded into the M (outer) dim of the CD tensor map, so the store's N coordinate
     must be group-**local** `blockIdx.x*BLOCK_N` (matches the verified SM100 kernel).
3. **Hand-built `GemmConfig` + exact smem** in the launcher instead of
   `get_best_config_asym`. The heuristic's CD-smem formula and FP32 swizzle choice do
   not match this kernel's actual layout, and it would enable 2-CTA multicast (which
   the kernel forbids). The launcher fixes `multicast=1`, `num_stages=2`,
   `block_n=128`, `block_k=128`, and computes the dynamic smem with the kernel's exact
   formula so the two can never disagree.
4. **TMA dtype mapping**: added `torch::kChar → CU_TENSOR_MAP_DATA_TYPE_UINT8` in
   `runtime_utils.hpp` (raw byte move; signedness is irrelevant to TMA).
5. **Scale-factor transpose done in Python dispatch** (not C++). The TMA SF
   descriptors read scales K-block-major / MN-contiguous, so the dispatch wrapper
   transposes `sfa [G,M,Kb]→[G,Kb,M]` and `sfb [G,N,Kb]→[Kb,G*N]` before the call.

### PCIe-resident expert weights (CUDA-pinned host memory)

The expert weights `B` may live in **CUDA-pinned host memory**; Hopper TMA fetches
them directly over the UVA address space (no manual copy to HBM). Only `B` may be
host-pinned — activations `A`, output `D`, and **both** scale tensors (`SFA`, `SFB`)
must stay in device memory. The host launcher asserts `b.is_cuda() or b.is_pinned()`
and requires `sfb.is_cuda()`. `runtime_utils.hpp` already builds `B`'s TMA descriptor
from `b.data_ptr()`, which is device-accessible for a pinned tensor under UVA.

Verified (`tests/test_h20_int8.py::test_int8_pinned_weights_sm90`): `max_diff = 0.0`
with `B` pinned-CPU and `SFB`/`A`/`D` on GPU.

Throughput (`tests/bench_h20_int8.py [--pinned]`, N=K=4096): because `B` is fetched
once per expert and reused across all `M` tokens, the PCIe cost amortizes as
tokens-per-expert grows:

| config        | B in HBM | B pinned-CPU |
|---------------|---------:|-------------:|
| G=8,  M=128   | 108 TOPS |     7 TOPS    |
| G=8,  M=512   | 152 TOPS |    28 TOPS    |
| G=32, M=2048  |  90 TOPS |    90 TOPS    |

So PCIe weight streaming is essentially free at large `M`-per-expert (decode with
high expert load / prefill) and PCIe-bound at small `M` (sparse decode).

### Current constraints (asserted in the host launcher)

* `K % 128 == 0` (SF granularity is fixed at 128, one scale per K-block).
* `N % 128 == 0` (`BLOCK_N = 128`); `M_max % 64 == 0` for masked.
* Output is FP32; A/B are `torch.int8` (`torch::kChar`).
* `BLOCK_N=192`/other valid int8 N would need a different `block_n`; not wired yet.
* `B` may be CUDA-pinned host memory (PCIe); `A`, `D`, `SFA`, `SFB` must be on GPU.

### Files actually touched

| File | Change |
|------|--------|
| `asym_gemm/common/sm90_utils.cuh` | `INT8MMA`, `INT8MMASelector`, `int32` fence overload |
| `asym_gemm/include/asym_gemm/impls/sm90_int8_asym_gemm_1d1d.cuh` | new kernel (+2 FP32 bug fixes) |
| `csrc/jit_kernels/impls/sm90_int8_asym_gemm_1d1d.hpp` | new launcher (masked + contiguous) |
| `csrc/jit_kernels/impls/runtime_utils.hpp` | int8 TMA dtype mapping |
| `csrc/apis/gemm.hpp` | `m_grouped_int8_asym_gemm_sm90_{masked,contiguous}` + pybind |
| `asym_gemm/utils/math.py` | `per_token_cast_to_int8`, `per_channel_cast_to_int8` |
| `asym_gemm/dispatch.py`, `asym_gemm/__init__.py` | int8 entry points + SF transpose |
| `tests/test_h20_int8.py` | INT8 correctness tests (smoke + masked + contiguous) |

The rest of this document is the original design plan, kept for reference.

---

This plan describes how to derive an **INT8** SM90 (Hopper / H20) asymmetric grouped
GEMM kernel from the existing **FP8** kernel
`asym_gemm/include/asym_gemm/impls/sm90_fp8_asym_gemm_1d1d.cuh`, wire it so it can
actually be launched, and verify its numerical correctness step by step until it
matches a float reference.

---

## 0. Current State (what already exists)

* `sm90_fp8_asym_gemm_1d1d.cuh` — a B-centric (outer-K, inner-M) WGMMA kernel that:
  * loads **B once per K-block** (single smem slot) and **A staged** (`kNumStages`),
  * loads **per-token FP32 SFA** (one float per A row, per K-block) and
    **per-channel FP32 SFB** (one float per B column, per K-block),
  * accumulates in **FP32 registers**, applies `scale_a * scale_b * accum` in the
    epilogue, writes FP32 to `smem_cd`, and K-reduces into HBM via
    `SM90_TMA_STORE_2D` (first K-block) / `SM90_TMA_REDUCE_ADD_2D` (later K-blocks).
* `asym_gemm/common/sm90_utils.cuh` — `FP8MMA` / `FP8MMASelector<N>` wrap the
  `MMA_64xNx32_F32E4M3E4M3_SS_TN` cute atoms (M=64, K=32, FP32 accumulators).
* **Important:** the SM90 FP8 1D1D kernel is *not yet wired into dispatch*. The
  unified `m_grouped_fp8_asym_gemm_nt_*` API routes SM90 to the **SM89 native FP8
  MoE kernel** (`csrc/apis/gemm.hpp` → `m_grouped_fp8_asym_gemm_sm89*`). So the
  `.cuh` compiles but has no host launcher. `tests/test_h20_int8.py` is currently a
  byte-for-byte copy of `tests/test_h20_fp8.py` and exercises the FP8 path.
  See `summary_sm90.md` → "What Remains (Out of Scope)".

Consequence: to *test* an INT8 1D1D kernel we must add the host launcher + a Python
entry point that reaches the new kernel directly (the unified dispatcher only knows
bf16/fp8/fp4).

---

## 1. Goal & Scope

Produce `sm90_int8_asym_gemm_1d1d_impl` — identical control flow / pipeline / TMA /
barrier structure to the FP8 kernel, but:

| Aspect            | FP8 kernel                              | INT8 kernel                          |
|-------------------|-----------------------------------------|--------------------------------------|
| A/B element type  | `cutlass::float_e4m3_t` (1 byte)        | `int8_t` (1 byte)                    |
| WGMMA atom        | `MMA_64xNx32_F32E4M3E4M3_SS_TN`         | `MMA_64xNx32_S32S8S8_SS_TN`         |
| Accumulator       | `float accum[]` (FP32)                  | `int32_t accum[]` (S32)             |
| K-atom            | 32                                      | 32 (unchanged)                       |
| Scale apply       | `scale_a*scale_b*accum`                 | `scale_a*scale_b*float(accum)`       |
| Output (C/D)      | FP32 (enforced)                         | FP32 (enforced)                      |
| Quant scheme      | per-token A / per-channel B, FP32 SF    | same (per-token A / per-channel B)   |

Because `sizeof(int8_t) == sizeof(float_e4m3_t) == 1`, **all shared-memory sizes,
swizzle modes, TMA descriptors and GMMA byte-offset math are unchanged**. The only
real differences are the MMA atom, the accumulator type, and the `int32 → float`
cast in the epilogue.

Deliverables:
1. `asym_gemm/common/sm90_utils.cuh` — add `INT8MMA` + `INT8MMASelector<N>` and an
   `int32_t` overload of `warpgroup_fence_operand`.
2. `asym_gemm/include/asym_gemm/impls/sm90_int8_asym_gemm_1d1d.cuh` — the kernel.
3. `csrc/jit_kernels/impls/sm90_int8_asym_gemm_1d1d.hpp` — JIT host launcher
   (contiguous + masked), modeled on `sm100_fp8_asym_gemm_1d1d.hpp`.
4. `csrc/apis/gemm.hpp` + `register_apis` — `m_grouped_int8_asym_gemm_sm90*` pybind
   entry points.
5. `asym_gemm/utils/math.py` — `per_token_cast_to_int8`, `per_channel_cast_to_int8`.
6. `asym_gemm/__init__.py` / `asym_gemm/dispatch.py` — export the int8 entry points.
7. `tests/test_h20_int8.py` — real INT8 correctness tests (self-contained, see §6).

---

## 2. `sm90_utils.cuh` — INT8 MMA wrapper

Add directly after `FP8MMASelector` (mirror it). Note the cute fma for int8 takes
`uint32_t&` accumulator registers, so we reinterpret the `int32_t` storage.

```cpp
template <int N_, typename MMA>
struct INT8MMA {
    template <size_t ...Idx>
    __forceinline__ __device__ static void
    call_fma_impl(uint64_t const& desc_a, uint64_t const& desc_b,
                  int32_t* d, bool scale_d, cute::index_sequence<Idx...>) {
        using namespace cute::SM90::GMMA;
        MMA::fma(desc_a, desc_b,
                 reinterpret_cast<uint32_t&>(d[Idx])...,
                 (scale_d ? ScaleOut::One : ScaleOut::Zero));
    }
    __forceinline__ __device__ static void
    wgmma(uint64_t const& desc_a, uint64_t const& desc_b, int32_t* d, bool scale_d) {
        call_fma_impl(desc_a, desc_b, d, scale_d, cute::make_index_sequence<N_/2>{});
    }
    static constexpr int M = 64;
    static constexpr int N = N_;
    static constexpr int K = 32;
    static constexpr int kNumAccum = M * N / 128;   // == N/2, same as FP8
};

template <int N>
struct INT8MMASelector {
    static constexpr auto select_mma() {
        using namespace cute::SM90::GMMA;
        // INT8 GMMA supports N ∈ {8,16,24,32,48,64,80,96,112,128,144,160,
        //                          176,192,208,224,240,256}
        // (NOTE: unlike FP8 it does NOT provide 40/56/72/.../248 — BLOCK_N
        //  must be one of the values above; 128 is the default and is valid.)
        if constexpr (N == 8)   return MMA_64x8x32_S32S8S8_SS_TN();
        if constexpr (N == 16)  return MMA_64x16x32_S32S8S8_SS_TN();
        if constexpr (N == 24)  return MMA_64x24x32_S32S8S8_SS_TN();
        if constexpr (N == 32)  return MMA_64x32x32_S32S8S8_SS_TN();
        if constexpr (N == 48)  return MMA_64x48x32_S32S8S8_SS_TN();
        if constexpr (N == 64)  return MMA_64x64x32_S32S8S8_SS_TN();
        if constexpr (N == 80)  return MMA_64x80x32_S32S8S8_SS_TN();
        if constexpr (N == 96)  return MMA_64x96x32_S32S8S8_SS_TN();
        if constexpr (N == 112) return MMA_64x112x32_S32S8S8_SS_TN();
        if constexpr (N == 128) return MMA_64x128x32_S32S8S8_SS_TN();
        if constexpr (N == 144) return MMA_64x144x32_S32S8S8_SS_TN();
        if constexpr (N == 160) return MMA_64x160x32_S32S8S8_SS_TN();
        if constexpr (N == 176) return MMA_64x176x32_S32S8S8_SS_TN();
        if constexpr (N == 192) return MMA_64x192x32_S32S8S8_SS_TN();
        if constexpr (N == 208) return MMA_64x208x32_S32S8S8_SS_TN();
        if constexpr (N == 224) return MMA_64x224x32_S32S8S8_SS_TN();
        if constexpr (N == 240) return MMA_64x240x32_S32S8S8_SS_TN();
        if constexpr (N == 256) return MMA_64x256x32_S32S8S8_SS_TN();
    }
    static constexpr auto select_type() { return INT8MMA<N, decltype(select_mma())>(); }
    using type = decltype(select_type());
};
```

Add the int32 fence overload next to the existing float one:

```cpp
__forceinline__ __device__ void warpgroup_fence_operand(int32_t& reg) {
    asm volatile("" : "+r"(reg) :: "memory");
}
```

(The S32S8S8 atoms are present in `third-party/cutlass/include/cute/arch/
mma_sm90_gmma.hpp`; confirmed in the bundled DeepGEMM cutlass copy.)

---

## 3. `sm90_int8_asym_gemm_1d1d.cuh` — the kernel

Copy `sm90_fp8_asym_gemm_1d1d.cuh` to the new file and apply these mechanical edits.
Line numbers refer to the FP8 file.

1. **Name + namespace use** — rename function to `sm90_int8_asym_gemm_1d1d_impl`.
   Keep `using namespace asym_gemm::sm90;`.

2. **WGMMA type (line 57)**
   ```cpp
   using WGMMA = typename INT8MMASelector<BLOCK_N>::type;
   ```

3. **Element type** — replace every `cutlass::float_e4m3_t` with `int8_t`:
   * `SMEM_A_SIZE_PER_STAGE`, `SMEM_B_SIZE_PER_STAGE` (lines 113–114) — size is
     identical (1 byte) but use `sizeof(int8_t)` for clarity.
   * `smem_a` / `smem_b` reinterpret casts (lines 133, 136).
   * `tma_copy<..., int8_t, ...>` for A and B (lines 223, 226, 261, 264).
   * `make_gmma_desc<..., int8_t? no>` — `make_gmma_desc` does not take the element
     type; only `advance_gmma_desc_lo<..., cutlass::float_e4m3_t>` does
     (lines 390, 392). Replace its element-type template arg with `int8_t`.

4. **Static assert (line 61)** — keep "INT8 asym GEMM on SM90 requires FP32 output";
   keep `BLOCK_K % WGMMA::K == 0` (K=32, unchanged). The valid INT8 `BLOCK_N` set is
   `{8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256}` — note this is
   *narrower* than FP8 (FP8 also has 40/56/72/…/248). The `INT8MMASelector`
   `if constexpr` chain is the real guard: an unsupported `BLOCK_N` falls through to a
   `void`-returning `select_mma()` and fails to compile, so no separate assert is
   strictly required (the FP8 default `BLOCK_N=128` is valid for INT8).

5. **Accumulator (line 351)**
   ```cpp
   int32_t accum[WGMMA::kNumAccum * kNumMWaves] = {0};
   ```
   `WGMMA::wgmma(a_desc, b_desc, shifted_accum, 1)` now takes `int32_t*` (line 394).

6. **Fences (lines 381–382, 398–400)** — already overloaded for `int32_t&` in §2.
   No code change beyond `accum` now being `int32_t`.

7. **Epilogue scale application (lines 431–434)** — cast the int32 accumulator to
   float before scaling:
   ```cpp
   float val_0 = scale_a_0[local_idx] * scales_b[i].x * static_cast<float>(shifted_accum[i*4+0]);
   float val_1 = scale_a_0[local_idx] * scales_b[i].y * static_cast<float>(shifted_accum[i*4+1]);
   float val_2 = scale_a_1[local_idx] * scales_b[i].x * static_cast<float>(shifted_accum[i*4+2]);
   float val_3 = scale_a_1[local_idx] * scales_b[i].y * static_cast<float>(shifted_accum[i*4+3]);
   ```
   SFA/SFB stay **FP32** (`smem_sfa`/`smem_sfb` unchanged) — they are the dequant
   scales `amax/127` for A and B, not the data.

8. **Everything else is unchanged**: scheduler, barriers, pipeline phases, B single
   slot, SFA/SFB TMA loads, `smem_cd` (FP32), `SM90_TMA_STORE_2D` /
   `SM90_TMA_REDUCE_ADD_2D` K-reduction, double-buffered TMA store, the WAR-hazard
   ordering (read scales *before* `empty_barrier_arrive_a`).

The register layout the epilogue assumes (`r_0`, `r_1`, `col_idx`, `scales_b`
indexing) is identical for S32 and F32 WGMMA accumulators (both 64×N, 2 regs per
thread per 8 columns), so no index math changes.

---

## 4. Host launcher `csrc/jit_kernels/impls/sm90_int8_asym_gemm_1d1d.hpp`

Model on `sm100_fp8_asym_gemm_1d1d.hpp`. Two classes
(`SM90Int8AsymGemm1D1DRuntime`, `SM90Int8AsymGemmMaskedRuntime`) whose
`generate_impl` emits `#include <asym_gemm/impls/sm90_int8_asym_gemm_1d1d.cuh>` and
instantiates `sm90_int8_asym_gemm_1d1d_impl<...>` with the same 22 template params.

Key differences vs SM100:
* Use the **SM90** arch spec / heuristic when picking `block_m/n/k`, threads,
  stages, swizzle, multicast. A safe starting **manual** config (matching the FP8
  test path) is `BLOCK_M=128, BLOCK_N=128, BLOCK_K=128`, `kNumMulticast=1`,
  `kIsMulticastOnA` per spec, output `cd_dtype = float`.
* `cd_dtype_t = float` (INT8 output is FP32, like FP8).
* The A/B TMA descriptors use **1-byte** element type (same as FP8) — reuse
  `make_tma_a_desc`/`make_tma_b_desc` with the int8 tensors.
* SFA/SFB TMA descriptors: FP32, per-token (A) / per-channel (B), one scale per
  128-element K-block — same layout the FP8 kernel expects.
* Grid: `{ceil_div(n, BLOCK_N), num_groups}`; masked early-exit handled in-kernel.

Provide two host functions:
```cpp
sm90_m_grouped_int8_asym_gemm_contiguous_1d1d(a, sfa, b, sfb, d, offsets, experts,
                                              grid_y, num_groups, m, n, k,
                                              major_a, major_b, compiled_dims);
sm90_m_grouped_int8_asym_gemm_masked_1d1d(a, sfa, b, sfb, d, masked_m, expected_m,
                                          num_groups, m, n, k,
                                          major_a, major_b, compiled_dims);
```

---

## 5. Python API surface

`csrc/apis/gemm.hpp`:
* Add `m_grouped_int8_asym_gemm_sm90(...)` (contiguous) and
  `m_grouped_int8_asym_gemm_sm90_masked(...)` host wrappers that validate shapes
  (`a/b` int8, `d` float32, `sfa/sfb` float32, K%32==0, N a valid GMMA N), then call
  the launchers in §4. Guard with `#if DG_TENSORMAP_COMPATIBLE` (needs TMA).
* Register both in `register_apis`.

`asym_gemm/__init__.py`: add the two names to `_maybe_import_from_C([...])`.

`asym_gemm/dispatch.py`: add architecture-agnostic wrappers
`m_grouped_int8_asym_gemm_nt_contiguous` / `_nt_masked` that accept `(data, scale)`
pairs and, on SM90, call the new `_C.m_grouped_int8_asym_gemm_sm90*`. (No SM100 int8
path exists; raise a clear error there for now.) Export from `__init__`.

`asym_gemm/utils/math.py` — INT8 quantizers (round-to-nearest, clamp ±127):
```python
def per_token_cast_to_int8(x, gran_k=128):       # x: [M, K] -> (int8 [M,K], fp32 sf [M, K//gran_k])
    m, n = x.shape
    padded_n = align(n, gran_k)
    xp = x.new_zeros((m, padded_n)); xp[:, :n] = x
    xv = xp.view(m, -1, gran_k)
    amax = xv.abs().float().amax(dim=2).clamp(1e-4)
    sf = amax / 127.0
    q = (xv / sf.unsqueeze(2)).round().clamp(-127, 127).to(torch.int8)
    return q.view(m, padded_n)[:, :n].contiguous(), sf

def per_channel_cast_to_int8(x, gran_k=128):     # x: [N, K] (per-row==per-channel of B^T)
    # same as per_token but documents B's per-output-channel scaling
    return per_token_cast_to_int8(x, gran_k)
```
(`sf` is `amax/127` because INT8 range is [-127, 127]; symmetric, no zero-point.)

---

## 6. Test file `tests/test_h20_int8.py`

Rewrite it as a **self-contained** INT8 test (do not depend on the FP8 generators).
It builds int8 inputs + FP32 scales, computes a float reference, calls the new int8
API, and compares with `calc_diff`. It must:

* `@pytest.mark.skipif(get_arch_major() != 9, ...)` (SM90 only).
* Skip gracefully if the int8 entry point isn't built yet, so the file is valid
  before the kernel lands (`pytest.importorskip` style `getattr` guard).
* Include a **tiny smoke case** first (1 group, M=128, N=128, K=128) to localize
  bugs, then the full masked + contiguous sweeps.

Reference math (matches the kernel exactly: int32 accumulate per K-block, then
`SFA*SFB`, then FP32 K-reduction):
```python
ref = (a_q.float() * sfa_expand) @ (b_q.float() * sfb_expand).transpose(-1, -2)
```
where `sfa_expand`/`sfb_expand` broadcast each per-block scale across its 128 K
elements. Because each scale is constant within its K-block, dequantize-then-matmul
is numerically identical to the kernel's accumulate-then-scale.

The concrete test file is written alongside this plan (see the committed
`tests/test_h20_int8.py`). Its structure:
1. `quantize_a` / `quantize_b` helpers (per-token / per-channel, gran_k=128).
2. `ref_grouped(a_q, sfa, b_q, sfb, masked_m)` float reference.
3. `test_int8_smoke_sm90` — single group, fixed seed, assert `diff < 0.05`
   (int8 quant error is larger than fp8; threshold is looser — see §7).
4. `test_m_grouped_int8_masked_sm90` — sweep num_groups / n / k.
5. `test_m_grouped_int8_contiguous_sm90` — sweep with offsets/experts.

---

## 7. Verification process (iterative, until correct)

Run on an SM90 (H20/Hopper) box. Build with `pip install -e .` (JIT compiles the
kernel on first call). Proceed stage by stage; do **not** move to the next stage
until the current one passes.

### Stage A — compile only
* Goal: the new `.cuh` instantiates for SM90 without errors.
* How: add a throwaway host call (or the smoke test) so the JIT triggers; or
  `nvcc -arch=sm_90a -c` a TU that includes the header and instantiates
  `sm90_int8_asym_gemm_1d1d_impl<...>` with the manual config.
* Pass: no compile errors. Common failures: wrong MMA atom name, passing `float*`
  to `INT8MMA::wgmma`, missing `int32_t` fence overload, BLOCK_N not a valid int8 N.

### Stage B — numeric smoke (single group, no K-reduction surprises)
* Run `test_int8_smoke_sm90` with `random.seed(0); torch.manual_seed(0)`.
* Use K=128 (a single K-block) first so there is **no** `REDUCE_ADD` path; this
  isolates the MMA + scale-apply + single `STORE_2D`.
* Pass: `calc_diff(d_kernel, ref) < 0.05`. Inspect a few elements:
  `print(d_kernel[0,:4], ref[0,:4])`. If off by a constant factor → scale wiring
  (SFA/SFB swapped, or `/127` vs `/255`). If random garbage → MMA layout / desc.

### Stage C — multi-K-block (exercise REDUCE_ADD)
* K=256 / 512 so `block_k` iterates >1 and `SM90_TMA_REDUCE_ADD_2D` runs.
* Pass: `diff < 0.05`. If first K-block is right but result grows/saturates →
  store-vs-reduce ordering (`block_k_iter == 0` must `STORE`, else `REDUCE_ADD`).

### Stage D — masked grouped sweep
* `test_m_grouped_int8_masked_sm90` over `num_groups ∈ {1,4,8}`, `n,k` sweep,
  `masked_m` drawn ±30% around `expected_m` (reuse the FP8 generator pattern). For
  each group compare only `d[j, :masked_m[j]]`.
* Pass: `max_diff < 0.05` for every group. Watch the `m_end==0` early-exit and the
  per-group `m_idx`/`local_m_idx` indexing (identical to FP8).

### Stage E — contiguous grouped sweep
* `test_m_grouped_int8_contiguous_sm90` with offsets/experts built by
  `build_offsets_experts_from_m_indices_pairs` (BLOCK_M=128 alignment). Mask out
  `m_indices == -1` rows before comparing (as the FP8 test does).
* Pass: `diff < 0.05`.

### Stage F — robustness / edge cases
* N not a multiple of 128 (but a valid int8 GMMA N, e.g. 64, 192); K a multiple of
  32 but not 128 (e.g. K=160) to test the last partial K-block scale indexing.
* Negative-heavy and saturating inputs (values that hit ±127) to confirm clamp and
  no int32 overflow (K≤~32768 with int8 ±127 → max |accum| ≈ K·127² ≈ 5.3e8 < 2^31,
  safe; document this bound).
* Determinism: run twice, assert identical output (no atomics race in REDUCE_ADD
  beyond the intended K-reduction).

### Numerical-threshold rationale
INT8 (256 levels) is coarser than FP8-E4M3, so per-element relative error is larger.
`calc_diff` here is `1 - cosine_similarity`; a correct int8 GEMM against a float
reference typically lands well under `0.05`, often `< 0.01` for K≥512 (errors average
out). Start strict (`< 0.01`) and only loosen to `0.05` if a *correct* kernel
legitimately exceeds it for small K. A diff near `1.0` means structurally wrong, not
quantization noise.

### Debugging ladder (if a stage fails)
1. Replace SFA/SFB with all-ones and feed already-quantized int8 inputs whose true
   scale is 1 → kernel output should equal the **integer** matmul cast to float.
2. Set BLOCK_N = N = 64, M = 64 (one WGMMA tile, one warpgroup) to remove tiling.
3. Dump `accum` (int32) for thread 0 before scaling and compare to a hand-computed
   int32 dot product.
4. Bisect: FP8 kernel is known-good — diff the two `.cuh` files; the only deltas
   should be the 8 edits in §3.

---

## 8. Risk register

| Risk | Mitigation |
|------|------------|
| BLOCK_N chosen as 40/56/… (valid for FP8, invalid for INT8) | `INT8MMASelector` only lists legal N; assert in kernel + host. |
| Passing `float*` accum to int8 fma (compiles via implicit conv? no — `uint32_t&`) | `INT8MMA::wgmma` takes `int32_t*` and `reinterpret_cast<uint32_t&>`. |
| Forgetting the `int32_t` fence overload → wrong constraint `+f` on int reg | Add overload in §2; verify in Stage A. |
| int32 accumulator overflow for very large K | K·127² < 2^31 ⇒ K < ~133k; real K ≪ that. Documented in Stage F. |
| Scale granularity mismatch (kernel expects per-128-K block) | Quantizer uses gran_k=128; host SF TMA uses sf_quant_k=128. |
| SM90 kernel still unwired like FP8 | §4–§5 add a dedicated int8 launcher + API so the test can reach it. |

---

## 9. Summary of files to add / change

| File | Action |
|------|--------|
| `asym_gemm/common/sm90_utils.cuh` | add `INT8MMA`, `INT8MMASelector`, int32 `warpgroup_fence_operand` |
| `asym_gemm/include/asym_gemm/impls/sm90_int8_asym_gemm_1d1d.cuh` | **new** kernel (8 edits vs FP8) |
| `csrc/jit_kernels/impls/sm90_int8_asym_gemm_1d1d.hpp` | **new** host launcher (contiguous + masked) |
| `csrc/apis/gemm.hpp` | add `m_grouped_int8_asym_gemm_sm90*` + register |
| `asym_gemm/utils/math.py` | add `per_token_cast_to_int8`, `per_channel_cast_to_int8` |
| `asym_gemm/dispatch.py`, `asym_gemm/__init__.py` | export int8 entry points |
| `tests/test_h20_int8.py` | **rewrite** as real INT8 correctness tests (§6) |
</content>
</invoke>
