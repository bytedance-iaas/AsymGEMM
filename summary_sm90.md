# SM90 (H20/Hopper) Asymmetric GEMM — Implementation Summary

## Problem

`sm90_bf16_asym_gemm.cuh` and `sm90_fp8_asym_gemm_1d1d.cuh` were byte-for-byte copies of their SM100 (Blackwell/GB200) counterparts. They contained SM100-exclusive hardware constructs — Tensor Memory (TMEM), UMMA single-warp MMA, tcgen05 instructions — that do not exist on SM90 and therefore fail to compile for it.

---

## What "Asymmetric GEMM" Means

Standard GEMM tiles both A and B equally. **Asymmetric (B-centric)** GEMM is designed for workloads where B fits in SRAM but A is too large:

```
outer loop: K blocks         (B changes each K block)
  inner loop: M blocks       (A changes each M block)
    compute: C[m,n] += A[m,k] * B[k,n]
  K-reduction: TMA REDUCE ADD to global C (across K blocks)
```

B occupies a **single smem slot** (no staging), reused across all M blocks for a given K block. A uses a staged pipeline (kNumStages slots). K-reduction is done in HBM via `SM90_TMA_REDUCE_ADD_2D` (first K iteration uses `SM90_TMA_STORE_2D` to initialize).

---

## SM90 vs SM100 Architecture Differences

| Aspect | SM100 (Blackwell) | SM90 (Hopper) |
|---|---|---|
| MMA instruction | UMMA — issued by **1 warp** | WGMMA — issued by **128-thread warp-group** |
| Accumulator storage | **TMEM** (Tensor Memory, hardware register file) | **Registers** (`float accum[kNumAccum]`) |
| Scale-factor copy to TMEM | UTCCP transposer warp + tcgen05 | Not needed — no TMEM |
| Epilogue path | Separate epilogue warps read TMEM | Math warp-group reads its own registers |
| GMMA descriptor | `make_umma_desc` / `advance_umma_desc_lo` | `make_gmma_desc` / `advance_gmma_desc_lo` |
| Arch guard | `__CUDA_ARCH__ >= 1000` | `__CUDA_ARCH__ >= 900` |
| Namespace | `asym_gemm::sm100` | `asym_gemm::sm90` |

---

## Threading Model (SM90 Asym)

```
Total threads = kNumNonEpilogueThreads + kNumEpilogueThreads
             = kNumMathThreads         + kNumTMAThreads

warp_idx < kNumMathThreads/32   →  Math warp-group(s)
                                    • Issue WGMMA (warpgroup_arrive / commit / wait)
                                    • Write accum to smem_cd (STSM or st.shared)
                                    • Issue TMA STORE / TMA REDUCE ADD to HBM
                                    • Signal empty barriers for A and B

warp_idx >= kNumMathThreads/32  →  TMA warp-group
                                    • Load B (single slot) + SFB once per K block
                                    • Load A (staged)      + SFA once per M-block per K block
                                    • Initialize barriers (warp kNumMathThreads/32 + 1)
                                    • Prefetch TMA descriptors (warp kNumMathThreads/32)
```

The ABI keeps the SM100 template parameter names (`kNumNonEpilogueThreads`, `kNumEpilogueThreads`) so existing dispatch infrastructure does not require changes. Internally they are aliased to `kNumMathThreads` and `kNumTMAThreads`.

---

## Shared Memory Layout

```
[ smem_cd    : kNumTMAStoreStages × STORE_BLOCK_M × kSwizzleCDMode bytes ]
[ smem_a     : kNumStages         × LOAD_BLOCK_M  × BLOCK_K × sizeof(elem) ]
[ smem_b     : 1 slot             × LOAD_BLOCK_N  × BLOCK_K × sizeof(elem) ]
[ smem_sfa   : kNumStages         × BLOCK_M × sizeof(float)   ]  ← FP8 only
[ smem_sfb   : 1 slot             × BLOCK_N × sizeof(float)   ]  ← FP8 only
[ barriers   : full[kNumStages], empty[kNumStages], full_b[1], empty_b[1] ]
```

Barrier initialization counts:
- `full_barriers[i]->init(1)` — TMA fills one stage at a time
- `empty_barriers[i]->init(kNumMulticast * kNumMathThreads / 32)` — one arrival per math warp per CTA in cluster
- `full/empty_barriers_b`: same pattern for the single B slot

---

## WGMMA Pipeline (per M block)

```cpp
// Accumulators live in registers — zero-init per (K, M) tile
float accum[WGMMA::kNumAccum] = {0};

full_barriers[stage_idx]->wait(phase);           // A tile ready

// [FP8 only] Read SFA/SFB scales here, BEFORE releasing smem_a
//   scale_a_0/1 = smem_sfa[stage_idx][r_0/r_1]
//   scales_b[i] = smem_sfb[0][i*8 + col_idx*2]

warpgroup_fence_operand(accum);
warpgroup_arrive();
for k in 0..BLOCK_K/WGMMA::K:
    a_desc = advance_gmma_desc_lo<...>(a_base, wave_offset, k*WGMMA::K%BLOCK_ATOM_K, ...)
    b_desc = advance_gmma_desc_lo<...>(b_base, 0,           k*WGMMA::K%BLOCK_ATOM_K, ...)
    WGMMA::wgmma(a_desc, b_desc, accum, /*scale_d=*/k)
warpgroup_commit_batch();
warpgroup_fence_operand(accum);
warpgroup_wait<0>();

empty_barrier_arrive_a(stage_idx);              // release smem_a stage

// Epilogue: scale (FP8) → write to smem_cd → TMA store/reduce
```

The key ordering constraint for FP8: **scale reads must happen before `empty_barrier_arrive_a`** to avoid a WAR hazard where the TMA warp recycles `smem_sfa[stage_idx]` before the math side finishes reading it.

---

## FP8-Specific: Scale Factor Handling (1D1D)

Scale format: **float32**, per-token A scales and per-channel B scales.

- `SFA[BLOCK_M]`: one float per row of A — loaded once per M-block per K-block  
- `SFB[BLOCK_N]`: one float per column of B — loaded once per K-block (shared across M inner loop)
- Scale index: `sf_k_idx = block_k_iter` (one scale per 128-element K-block)

Scale application in epilogue (before writing to `smem_cd`):
```cpp
// WGMMA register layout: thread owns rows {r_0, r_1} and col pairs via col_idx
r_0 = wg_local_warp_idx * 16 + lane_idx / 4
r_1 = r_0 + 8
col_idx = lane_idx % 4

for i in 0..kNumAccum/4:
    final[i*4+0] = scale_a_0 * scales_b[i].x * accum[i*4+0]
    final[i*4+1] = scale_a_0 * scales_b[i].y * accum[i*4+1]
    final[i*4+2] = scale_a_1 * scales_b[i].x * accum[i*4+2]
    final[i*4+3] = scale_a_1 * scales_b[i].y * accum[i*4+3]
```

FP8 output is always FP32 (enforced by static assert).

---

## Epilogue: smem_cd → HBM

**BF16 output** (`cd_dtype_t = bfloat16_t`): uses `SM90_U32x2_STSM_N` (stmatrix.sync) with swizzled addresses to achieve coalesced bank-conflict-free writes.

**FP32 output** (always for FP8, optional for BF16): uses `st_shared` (plain store) into `smem_cd`.

After writing `smem_cd`, one thread issues TMA:
```cpp
if (block_k_iter == 0)
    SM90_TMA_STORE_2D::copy(...)           // write fresh result
else
    SM90_TMA_REDUCE_ADD_2D::copy(...)      // atomically add (K-reduction)
```

A double-buffer (`kNumTMAStoreStages = 2`) overlaps TMA stores across M blocks. `tma_store_wait<kNumTMAStoreStages - 1>()` is called before reusing a smem_cd slot.

---

## Static Constraints (SM90 Asym)

```cpp
DG_STATIC_ASSERT(kNumMWaves == 1,
    "BLOCK_M > WAVE_BLOCK_M not supported for SM90 asym GEMM");
// Why: smem_cd is sized for WAVE_BLOCK_M only; kNumMWaves>1 would overrun it.

DG_STATIC_ASSERT(kNumMulticast == 1 or kIsMulticastOnA,
    "B-side multicast not supported in SM90 asym GEMM");
// Why: n_idx is fixed per CTA; no n_idx adjustment is made for cluster peers.
```

---

## Files Changed

| File | Change |
|---|---|
| `asym_gemm/include/asym_gemm/impls/sm90_bf16_asym_gemm.cuh` | Full rewrite for SM90 |
| `asym_gemm/include/asym_gemm/impls/sm90_fp8_asym_gemm_1d1d.cuh` | Full rewrite for SM90 |
| `tests/test_h20_bf16.py` | New — SM90-gated BF16 asym GEMM tests |
| `tests/test_h20_fp8.py` | New — SM90-gated FP8 1D1D asym GEMM tests |

---

## Commit History

```
2faf2a1  fix: correct offset format and recipe in SM90 test files
fc77283  fix: add recipe=(1, 1, 128) to masked FP8 asym GEMM test call
43d1b60  test: add SM90 (H20) FP8 1D1D asym GEMM test cases
5af755d  test: add SM90 (H20) BF16 asym GEMM test cases
e3ea41c  fix: move SFA reads before empty_barrier in SM90 FP8 asym epilogue
f00c585  feat: rewrite sm90_fp8_asym_gemm_1d1d.cuh for SM90 architecture
1cd28d9  fix: add defensive static asserts for unsupported SM90 asym GEMM configs
276da2d  feat: rewrite sm90_bf16_asym_gemm.cuh for SM90 (H20/Hopper) architecture
```

---

## What Remains (Out of Scope)

The dispatch infrastructure (`csrc/apis/gemm.hpp` or equivalent) needs to route SM90 GPUs to these kernels at runtime. The kernels compile for SM90 but cannot be invoked until that routing is wired. Test files will skip on non-SM90 hardware with `pytest.mark.skipif(get_arch_major() != 9, ...)`.
