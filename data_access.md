# SM80 MoE FP8 GEMM — Vectorised Output Store Plan (`data_access.md`)

## 1. Problem: Element-by-Element Global Store in `write_output`

### Current code (sm80_moe_gemm.cuh, `write_output` lambda, lines ~465-487)

```
Stage 1  rO (BF16 regs) ──smem_copy_O──► sO (BF16 smem)       [vectorised – OK]
Stage 2  sO (smem)       ──gmem_copy_o──► tOrO (BF16 regs)     [128-bit LDS – OK]
Stage 3  tOrO (regs)     ──scalar loop──► tOgO (global mem)     [SLOW – scalar stores]
```

Stage 3 is the problem:

```cpp
cute::copy(gmem_tiled_copy_o, tOsO_src, tOrO);   // stage 2 – vectorised LDS.128 ✓
for (int mi = 0; mi < size<1>(tOgO); mi++) {
    int m_coord = get<0>(tOcO(_0{}, mi, _0{}));
    if (m_coord < m_actual) {
        for (int ai = 0; ai < size<0>(tOgO); ai++)
            for (int ni = 0; ni < size<2>(tOgO); ni++)
                tOgO(ai, mi, ni) = tOrO(ai, mi, ni);  // stage 3 – scalar STG ✗
    }
}
```

`GmemTiledCopyO` is configured with `Copy_Atom<UniversalCopy<uint128_t>, ElementOut>` and
a `Layout<Shape<_1, _8>>` value layout — meaning each thread's copy atom can issue one
**128-bit store (8 BF16)** per call.  The element-by-element assignment `tOgO(ai, mi, ni) =
tOrO(ai, mi, ni)` never invokes that atom; it writes one BF16 at a time through CuTe's tensor
element accessor, producing scalar `STG.32` or `STG.16` instructions instead.

**Impact:** 8× fewer bytes per instruction.  For a 128×128 BF16 output tile, stage 3 issues
128×128 = 16 384 scalar stores instead of 2 048 vectorised ones.  This dominates write
latency for every M-tile.

The same issue exists in `sm80_moe_gemm_impl` (BF16/FP16 kernel), in the identical
element-by-element triple loop at lines ~307-316.

---

## 2. Reference: How FA2 and mixtureExpertKernel Solve This

### 2.1 FA2 — `FLASH_NAMESPACE::copy` helper (`utils.h`)

```cpp
template <bool Is_even_MN=true, bool Is_even_K=true,
          bool Clear_OOB_MN=false, bool Clear_OOB_K=true,
          typename TiledCopy, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1,
          typename Engine2, typename Layout2,
          typename Engine3, typename Layout3>
__forceinline__ __device__ void copy(
    TiledCopy tiled_copy,
    Tensor<Engine0, Layout0> const &S,   // source  (tOrO or tVgV)
    Tensor<Engine1, Layout1>       &D,   // dest    (tOgO or tVsV)
    Tensor<Engine2, Layout2> const &identity_MN,  // coordinate tensor (tOcO)
    Tensor<Engine3, Layout3> const &predicate_K,  // K-dim bool predicates (tOpO)
    const int max_MN = 0)                         // valid-row threshold
{
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || predicate_K(k)) {
                    cute::copy(tiled_copy, S(_, m, k), D(_, m, k)); // ← one copy atom call
                } else if (Clear_OOB_K) {
                    cute::clear(D(_, m, k));
                }
            }
        } else if (Clear_OOB_MN) {
            cute::clear(D(_, m, _));
        }
    }
}
```

Key: `cute::copy(tiled_copy, S(_, m, k), D(_, m, k))` dispatches through the copy atom
(`UniversalCopy<uint128_t>` → `STG.128`), issuing one 128-bit store per logical tile per
thread.  Only the per-row predicate check (`get<0>(identity_MN(0, m, 0)) < max_MN`) gates
execution — no inner scalar loop.

FA2 output epilogue usage (`flash_fwd_kernel.h` lines ~119-121):

```cpp
// Full tile path (Is_even_MN inferred at compile time):
FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
    gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO,
    binfo.actual_seqlen_q - m_block * kBlockM);
```

A single call handles both full and partial tiles by branching at compile time on
`Is_even_MN` and at runtime on the row coordinate.

### 2.2 `mixtureExpertKernel.cu` — same pattern, applied to MoE

The reference kernel in this repo uses the identical approach for the O store:

```cpp
// Full tile:
FLASH_NAMESPACE::copy</*Is_even_MN=*/true, /*Is_even_K=*/true>(
    gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO);

// Partial tile (last M-tile, gap = remaining valid rows):
int gap = len - kBlockM * m_max + kBlockM;
FLASH_NAMESPACE::copy</*Is_even_MN=*/false, /*Is_even_K=*/true,
                      /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
    gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, gap);
```

This produces vectorised `STG.128` for every valid row, with zero scalar fallback.

---

## 3. Proposed Fix

### 3.1 Add a local predicated-copy helper

`sm80_moe_gemm.cuh` does not include FA2 headers.  Rather than adding that dependency, add
a self-contained inline helper directly in the file, before both kernel templates:

```cpp
// ─────────────────────────────────────────────────────────────────────────────
// moe_predicated_copy: vectorised tile copy with optional M-row predication.
//
// Template params:
//   Is_even_MN — if true, all rows are valid; skip coordinate check (fast path).
//   Is_even_K  — if true, all K columns are valid; skip predicate_K check.
//   Clear_OOB_MN — if true, clear destination rows that are out-of-bounds.
//   Clear_OOB_K  — if true, clear destination columns that are out-of-bounds.
//
// Parameters:
//   tiled_copy   — TiledCopy object (must wrap a 128-bit atom for best perf).
//   S            — source tensor, rank-3: (Atom, MMA_M, MMA_N).
//   D            — destination tensor, same shape as S.
//   identity_MN  — coordinate tensor (partition of make_identity_tensor for the
//                  MN tile); identity_MN(0, m, 0) gives the M-row index for row m.
//   predicate_K  — bool tensor of length size<2>(S); predicate_K(k) == true iff
//                  column k is within bounds.
//   max_MN       — exclusive upper bound on valid M rows (ignored when Is_even_MN).
// ─────────────────────────────────────────────────────────────────────────────
template <bool Is_even_MN = true, bool Is_even_K = true,
          bool Clear_OOB_MN = false, bool Clear_OOB_K = false,
          typename TiledCopy,
          typename Engine0, typename Layout0,  // source
          typename Engine1, typename Layout1,  // dest
          typename Engine2, typename Layout2,  // identity_MN
          typename Engine3, typename Layout3>  // predicate_K
CUTE_DEVICE void moe_predicated_copy(
    TiledCopy tiled_copy,
    cute::Tensor<Engine0, Layout0> const& S,
    cute::Tensor<Engine1, Layout1>&       D,
    cute::Tensor<Engine2, Layout2> const& identity_MN,
    cute::Tensor<Engine3, Layout3> const& predicate_K,
    int max_MN = 0)
{
    static_assert(!(Clear_OOB_MN && !Clear_OOB_K),
                  "Clear_OOB_MN requires Clear_OOB_K");
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(S); ++m) {
        if (Is_even_MN || cute::get<0>(identity_MN(0, m, 0)) < max_MN) {
            CUTE_UNROLL
            for (int k = 0; k < cute::size<2>(S); ++k) {
                if (Is_even_K || predicate_K(k))
                    cute::copy(tiled_copy, S(cute::_, m, k), D(cute::_, m, k));
                else if (Clear_OOB_K)
                    cute::clear(D(cute::_, m, k));
            }
        } else if (Clear_OOB_MN) {
            cute::clear(D(cute::_, m, cute::_));
        }
    }
}
```

This is a verbatim port of `FLASH_NAMESPACE::copy` with `cute::` namespace prefixes
instead of relying on `using namespace cute`.

### 3.2 Hoist coordinate and K-predicate tensors outside `write_output`

Currently `tOcO` and the temporary `tOrO` are reconstructed inside `write_output` on every
call.  These depend only on `gmem_thr_copy_o`, `cO`, `sO`, and `gO_m` — all of which are
captured by reference and stable across the M-tile loop.

Move the coordinate tensor construction **above** the `write_output` lambda definition:

```cpp
// ── Hoist: coordinate tensor for M-predication (output side) ────────────────
// tOcO maps each thread's copy partition to (M-row, N-col) coordinates.
// partition_D matches the destination (gO_m) layout; since cO is an identity
// tensor its values are the same as partition_S.
Tensor tOcO = gmem_thr_copy_o.partition_D(cO);

// tOpO: K-column predicates for the O store.
// N is always a multiple of BLOCK_N (enforced at API boundary), so all N columns
// within a tile are valid. Fill once with true; Is_even_K=true skips the check.
Tensor tOpO = make_tensor<bool>(make_shape(cute::size<2>(
                  gmem_thr_copy_o.partition_D(
                      cute::make_identity_tensor(Shape<Int<BLOCK_M>, Int<BLOCK_N>>{})
                  ))));
cute::fill(tOpO, true);
```

### 3.3 Replace the element-by-element loop in `write_output`

**Old stage 3 (inside the lambda):**

```cpp
cute::copy(gmem_tiled_copy_o, tOsO_src, tOrO);           // smem→reg (vectorised)
for (int mi = 0; mi < size<1>(tOgO); mi++) {              // ← scalar loop
    int m_coord = get<0>(tOcO(_0{}, mi, _0{}));
    if (m_coord < m_actual) {
        for (int ai = 0; ai < size<0>(tOgO); ai++)
            for (int ni = 0; ni < size<2>(tOgO); ni++)
                tOgO(ai, mi, ni) = tOrO(ai, mi, ni);      // ← scalar STG
    }
}
```

**New stage 3:**

```cpp
cute::copy(gmem_tiled_copy_o, tOsO_src, tOrO);   // smem→reg (vectorised LDS.128)

if (m_actual == static_cast<int>(BLOCK_M)) {
    // Full tile: all rows valid — compile-time fast path, no predicate overhead.
    moe_predicated_copy</*Is_even_MN=*/true, /*Is_even_K=*/true>(
        gmem_tiled_copy_o, tOrO, tOgO, tOcO, tOpO);
} else {
    // Partial last tile: skip OOB rows, vectorised store for valid rows.
    // Clear_OOB_MN=false: do not write zeros to gmem for rows >= m_actual.
    moe_predicated_copy</*Is_even_MN=*/false, /*Is_even_K=*/true,
                        /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_o, tOrO, tOgO, tOcO, tOpO, m_actual);
}
```

`cute::copy(tiled_copy, S(_, m, k), D(_, m, k))` inside `moe_predicated_copy` dispatches
through `UniversalCopy<uint128_t>`, issuing one `STG.128` (128-bit = 8 BF16) per thread per
tile instead of 8 scalar stores.

### 3.4 Apply the same fix to `sm80_moe_gemm_impl` (BF16/FP16 kernel)

`sm80_moe_gemm_impl` has the identical pattern in the output write section (lines ~307-316).
Apply the same three changes: hoist `tOcO`/`tOpO`, replace the triple loop with
`moe_predicated_copy`, split on `m_actual == BLOCK_M`.

---

## 4. Files to Change

| File | Changes |
|------|---------|
| `asym_gemm/include/asym_gemm/impls/sm80_moe_gemm.cuh` | Add `moe_predicated_copy` helper before both kernels; fix `sm80_moe_fp8_gemm_impl` `write_output`; fix `sm80_moe_gemm_impl` output loop |

No other files require changes.

---

## 5. Expected Performance Gain

The store path per M-tile changes from:

| Path | Instructions issued | Bytes per instruction |
|------|--------------------|-----------------------|
| Old (scalar loop) | `BLOCK_M × BLOCK_N` scalar stores | 2 (BF16) or 4 (FP32) |
| New (vectorised) | `BLOCK_M × BLOCK_N / 8` 128-bit stores | 16 |

For `BLOCK_M=128, BLOCK_N=128`: 16 384 → 2 048 store instructions per M-tile.  The
improvement is most visible when store latency is on the critical path (low arithmetic
intensity, small token counts) and for kernels with many M-tiles (large total_tokens).

---

## 6. Correctness Verification

### 6.1 What must not change

- Numeric output: `tOgO(ai, mi, ni)` after the fix must contain the same values as before.
- Partial-tile boundary: rows `>= m_actual` must not be written to gmem.
  `Clear_OOB_MN=false` guarantees this — `moe_predicated_copy` simply skips those rows.

### 6.2 Why correctness is preserved

1. **Full tile** (`Is_even_MN=true`): the `m < max_MN` check is compiled away; every row
   calls `cute::copy(tiled_copy, S(_, m, k), D(_, m, k))` which is the same data movement
   as the old `tOgO(ai, mi, ni) = tOrO(ai, mi, ni)` loop, just in one 128-bit instruction.

2. **Partial tile** (`Is_even_MN=false`): `get<0>(identity_MN(0, m, 0)) < m_actual` is the
   same predicate as the old `m_coord < m_actual`.  Valid rows are copied atomically;
   invalid rows are skipped (not zeroed), matching the old behaviour.

3. **K columns**: `Is_even_K=true` skips the K predicate — equivalent to the old code which
   had no K predicate at all (N is always BLOCK_N-aligned).

### 6.3 Test plan

Run the existing test suite with **no tolerance changes**:

```bash
python tests/test_sm80_moe.py
```

- All correctness cases must pass with `diff < 0.01` (same threshold as before).
- Benchmark output (TFLOPS) should improve for most shapes, particularly decode-phase
  cases with small token counts where store latency dominates.

Specific edge cases to verify:
- `token_counts = [7, 13, 31, 127]` — exercises partial M-tiles
- `token_counts = [512]` — single expert, full tiles only
- `list_size = 1` — exercises the `expert_e == 0` branch in the parallelism change
- All existing `TEST_CASES` in `test_sm80_moe.py`
