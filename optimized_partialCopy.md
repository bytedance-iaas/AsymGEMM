# SM80 MoE GEMM — Vectorised Async Partial-Tile Copy (`optimized_partialCopy.md`)

## 1. Problem: Synchronous Scalar Copy in the Partial-X-Tile Path

When `m_actual < BLOCK_M` (the last M-tile of an expert with non-multiple-of-`BLOCK_M`
token count), the kernel falls off the fast async path into a slow synchronous scalar
loop:

```cpp
if (m_actual == static_cast<int>(BLOCK_M)) {
    // FAST: vectorised cp.async, async, 16 FP8 (= 128-bit) per thread per atom
    cute::copy(gmem_tiled_copy_xw, tXgX_mk, tXsX);
    cp_async_fence();
    cp_async_wait<0>();
} else {
    // SLOW: drain pending cp.async, cooperatively zero sX, then per-element scalar copy
    cp_async_wait<0>();
    __syncthreads();
    clear_smem_region<BLOCK_M * BLOCK_K * sizeof(ElementIn)>(
        reinterpret_cast<char*>(smem_x), tidx, NWARPS * 32);
    __syncthreads();
    for (int mi = 0; mi < size<1>(tXsX); mi++) {
        int m_coord = get<0>(tXcX(_0{}, mi, _0{}));
        if (m_coord < m_actual)
            for (int ai = 0; ai < size<0>(tXsX); ai++)
                for (int ki = 0; ki < size<2>(tXsX); ki++)
                    tXsX(ai, mi, ki) = tXgX_mk(ai, mi, ki);   // ← scalar LDG + STS
    }
}
```

### Why the partial path is so much slower than the full path

The TiledCopy `gmem_tiled_copy_xw` is configured with
`Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, ElementIn>` and value layout
`<_1,_16>`. Going through the atom emits **one `cp.async.cg.shared.global.E16B`
instruction per thread per atom** — 16 FP8 elements (128 bits) per transaction,
asynchronous (overlaps with compute).

Going through the tensor element accessor (`tXsX(ai, mi, ki) = tXgX_mk(ai, mi, ki)`)
**bypasses the atom entirely**.  NVCC compiles that to one `LDG.U8` + one `STS.U8`
per FP8 element — 1 byte per instruction, synchronous, no overlap.

For the typical config (BLOCK_M=128, BLOCK_K=128, NWARPS=4, atom=16 FP8):

| Path | Instructions issued per row per thread | Bytes per instruction | Async? |
|------|----------------------------------------|------------------------|--------|
| Full (cp.async) | 2 (BLOCK_K / 16 atoms ÷ 4 threads-in-K) | 16 | yes |
| Partial (scalar) | 64 (LDG.U8 + STS.U8 per element) | 1 | no |

→ **32× fewer instructions** *plus* async/compute overlap on the fast path.

The partial path also wastes work on:
- A separate cooperative `clear_smem_region` pass (extra `__syncthreads` + STS).
- A `cp_async_wait<0>` immediately followed by an unnecessary `__syncthreads`,
  draining the pipeline before any X work even starts.

This branch is exercised on **every last M-tile of every expert** with a non-
multiple-of-`BLOCK_M` token count — pervasive in MoE workloads where token routing is
inherently irregular.

---

## 2. Reference: How FA2 and `mixtureExpertKernel.cu` Solve This

Both reference codebases use the **same predicated copy helper** for full and partial
tiles, with `Clear_OOB_MN=true` to handle OOB rows directly:

### FA2 — `flash_fwd_kernel.h:313-315` (V load with seq-len-tail predicate)

```cpp
// Clear the smem tiles to account for predicated off loads
FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
    gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV,
    binfo.actual_seqlen_k - n_block * kBlockN);
```

### `mixtureExpertKernel.cu:218-221` — same MoE workload, partial X load

```cpp
int gap = len - kBlockM * m_max + kBlockM;
FLASH_NAMESPACE::copy</*Is_even_MN=*/false, /*Is_even_K=*/true, /*Clear_OOB_MN=*/true>(
    gmem_tiled_copy_QKV, tXgX(_, _, _, m, k), tXsX, tXcX, tXpX, gap);
```

### What the helper does inside (`utils.h:296-361`)

```cpp
for (int m = 0; m < size<1>(S); ++m) {
    if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
        for (int k = 0; k < size<2>(S); ++k) {
            cute::copy(tiled_copy, S(_, m, k), D(_, m, k));   // ← cp.async via atom
        }
    } else if (Clear_OOB_MN) {
        cute::clear(D(_, m, _));                              // ← zero OOB row
    }
}
```

For valid rows: `cute::copy` dispatches through the cp.async atom — same vectorised
path the full-tile case takes.  
For OOB rows: `cute::clear` writes zeros to the row's smem (a few scalar STS, but only
for the small number of OOB rows — and those rows are still owned by exactly one
thread, so no cooperative loop is needed).

After the helper, a single `cp_async_fence(); cp_async_wait<0>();` drains everything,
matching the full-tile path exactly.

---

## 3. Proposed Fix

### 3.1 Helper already exists

The local `moe_predicated_copy` helper added in `data_access.md` is already a verbatim
port of `FLASH_NAMESPACE::copy` and supports the required template parameters:

```cpp
template <bool Is_even_MN = true, bool Is_even_K = true,
          bool Clear_OOB_MN = false, bool Clear_OOB_K = false, ...>
CUTE_DEVICE void moe_predicated_copy(...);
```

The static-assert `!(Clear_OOB_MN && !Clear_OOB_K)` requires `Clear_OOB_K=true` when
`Clear_OOB_MN=true` (because `cute::clear(D(_, m, _))` zeros the entire row including
all K columns).  With `Is_even_K=true` the K predicate is never consulted, so
`Clear_OOB_K=true` is a no-op constraint — it just satisfies the invariant.

### 3.2 Replace each partial-X-copy block

For each of the three sites (BF16/FP16 K-loop, FP8 k=0, FP8 k>0), replace:

```cpp
} else {
    cp_async_wait<0>();
    __syncthreads();
    clear_smem_region<BLOCK_M * BLOCK_K * sizeof(ElementIn)>(
        reinterpret_cast<char*>(smem_x), tidx, NWARPS * 32);
    __syncthreads();
    for (int mi = 0; mi < size<1>(tXsX); mi++) {
        int m_coord = get<0>(tXcX(_0{}, mi, _0{}));
        if (m_coord < m_actual)
            for (int ai = 0; ai < size<0>(tXsX); ai++)
                for (int ki = 0; ki < size<2>(tXsX); ki++)
                    tXsX(ai, mi, ki) = tXgX_mk(ai, mi, ki);
    }
}
```

with:

```cpp
} else {
    // Partial last M-tile: same async cp.async path as full tile, but with per-row
    // M-predication.  OOB rows are zeroed in-place via Clear_OOB_MN=true.
    moe_predicated_copy</*Is_even_MN=*/false, /*Is_even_K=*/true,
                        /*Clear_OOB_MN=*/true,  /*Clear_OOB_K=*/true>(
        gmem_tiled_copy_xw, tXgX_mk, tXsX, tXcX, tXpX, m_actual);
    cp_async_fence();
    cp_async_wait<0>();
}
```

`tXpX` is the K-predicate vector (analogous to the `tOpO` we already build for the
output store).  Build it once above the M-tile loop, sized from `tXsX`'s K-mode and
filled with `true` (since `K % BLOCK_K == 0` is guaranteed by the API):

```cpp
Tensor tXpX = make_tensor<bool>(make_shape(size<2>(tXsX)));
cute::fill(tXpX, true);
```

### 3.3 What goes away

- The `clear_smem_region<BLOCK_M * BLOCK_K * sizeof(ElementIn)>(...)` cooperative
  zeroing call — replaced by per-OOB-row `cute::clear` inside the helper.
- One of the two `__syncthreads()` (the one between the clear and the scalar copy).
- The outer `cp_async_wait<0>()` that drained the W copy before X work — the new
  path issues X cp.async first, then a single `cp_async_fence(); cp_async_wait<0>();`
  drains both W and X together (matching the full-tile path).

The final `__syncthreads()` after the if/else stays — it's still needed to publish
the OOB-row sync clears (regular STS) to other threads.

`clear_smem_region` itself stays defined in the file (still used for `smem_x` in some
other paths if any; also good to keep as a utility).

---

## 4. Files to Change

| File | Change |
|------|--------|
| `asym_gemm/include/asym_gemm/impls/sm80_moe_gemm.cuh` | (a) Build `tXpX` once per kernel; (b) replace 3 partial-X-copy blocks with `moe_predicated_copy` + `cp_async_fence/wait` |

Single file, JIT-compiled — no `_C.so` rebuild required.

---

## 5. Sites to Edit

### Site ① — `sm80_moe_gemm_impl` (BF16/FP16), K-loop partial X copy (line ~339)

Old `else` block runs for every K-tile when this M-tile is the partial last tile.

### Site ② — `sm80_moe_fp8_gemm_impl`, k=0 path partial X copy (line ~630)

Runs once per partial last M-tile in the k=0 prologue.

### Site ③ — `sm80_moe_fp8_gemm_impl`, k>0 path partial X copy (line ~705)

Runs for every k>0 iteration on partial last M-tiles.  Highest-frequency site for
typical FFN shapes (k_max can be 32+).

In all three sites:
- `tXgX_mk` is the per-tile gmem source slice (already sliced for current k by the
  call site).
- `tXsX` and `tXcX` are partition_D and partition_S of `sX` and the identity tensor —
  identical names and semantics across all three sites.

---

## 6. Out of Scope (Follow-up)

Site ③ in the FP8 k>0 path also contains a **partial O read-back** that uses the
same scalar pattern but with `gmem_tiled_copy_o` (UniversalCopy<uint128_t>, not
cp.async):

```cpp
// Predicated O read-back into sO.
Tensor tOsO_d = gmem_thr_copy_o.partition_D(sO);
Tensor tOgO_r = gmem_thr_copy_o.partition_S(gO_m);
Tensor tOcO   = gmem_thr_copy_o.partition_S(cO);
for (int mi = 0; mi < size<1>(tOsO_d); mi++) {
    int m_coord = get<0>(tOcO(_0{}, mi, _0{}));
    for (int ai = 0; ai < size<0>(tOsO_d); ai++)
        for (int ni = 0; ni < size<2>(tOsO_d); ni++)
            tOsO_d(ai, mi, ni) = (m_coord < m_actual)
                ? tOgO_r(ai, mi, ni) : ElementOut(0.0f);
}
```

The same fix applies — `moe_predicated_copy<false, true, true, true>(gmem_tiled_copy_o, …)`
— but skips `cp_async_fence/wait` because UniversalCopy is synchronous.  Kept out of
this plan to keep the correctness review focused on the cp.async path.  Propose
handling it in a follow-up plan once this change is verified.

---

## 7. Expected Performance Gain

For each thread, per partial M-tile, per K-tile load:

| Metric | Old path | New path |
|--------|----------|----------|
| Instructions per valid row | 64 scalar (LDG.U8 + STS.U8 per FP8 elem) | 2 cp.async.E16B |
| Bytes per instruction | 1 | 16 |
| Async / overlap with compute? | no | yes |
| Cooperative clear pass | yes (extra `__syncthreads`) | no |

On top of the 32× instruction reduction, the cp.async path overlaps with the MMA
work that immediately follows the load — the synchronous scalar path forces the
thread to wait for every load to retire before issuing the next.

Workloads that benefit most:
- **MoE decode** (small, irregular per-expert token counts → most M-tiles are partial).
- **Many K-tiles** (e.g., FFN K=4096 → k_max=32 with BLOCK_K=128) — site ③ runs once
  per k per partial M-tile.
- **Many small experts** — a top-2 router with many experts each receiving tens of
  tokens hits the partial path on every expert.

For the largest-N tests in `test_sm80_moe.py` (e.g., `(16384, 4096, [64, 128, 32, 96])`)
this is the dominant remaining inefficiency in the load pipeline.

---

## 8. Correctness Verification

### 8.1 Why correctness is preserved

1. **Same partition, same coordinates.** `tXcX = gmem_thr_copy_xw.partition_S(cX)`
   gives each thread the same M-row coordinates as the old scalar loop. The check
   `get<0>(identity_MN(0, m, 0)) < max_MN` inside the helper is bit-identical to the
   old `m_coord < m_actual` check.
2. **Same valid-row data movement.** `cute::copy(tiled_copy, S(_, m, k), D(_, m, k))`
   for the cp.async atom moves the same `BLOCK_K`-worth of bytes per row that the old
   scalar triple-loop did, just in 128-bit transactions.
3. **OOB rows zeroed.** `Clear_OOB_MN=true` calls `cute::clear(D(_, m, _))` for OOB
   rows — same end state as `clear_smem_region` followed by skipping the row.
4. **Partition is exhaustive.** Every (M, K) position in `sX` is owned by exactly one
   thread for write — verified by the TiledCopy's thread×value layout covering
   `BLOCK_M × BLOCK_K`. So every OOB row gets cleared exactly once, and every valid
   row gets a cp.async exactly once.
5. **Memory ordering is preserved.** The new path issues cp.async (async) and
   per-row clear (sync STS) interleaved on each thread, then `cp_async_fence();
   cp_async_wait<0>(); __syncthreads()` drains both. Async writes commit on
   wait_group; sync writes are visible after `__syncthreads` (CTA-level membar).
   Same final visibility as the old path.

### 8.2 Test plan

```bash
python tests/test_sm80_moe.py
```

The test cases that exercise the partial-X path:
- `(8, 4096, 256, [12, 8, 20, 28])` — every expert has only partial M-tiles.
- `(8, 4096, 512, [128, 64, 256, 12])` — mix of full and partial.
- `(4, 4096, 256, [7, 13, 31, 127])` — heavily lopsided partials.
- `(8, 4096, 128, [100, 200, 50, 150])` — mid-size partials at K=128.

Expected: all 8 tests pass with diffs **bit-identical** to the current baseline
(0.000299 - 0.000872).  The cp.async path moves the same bytes as the scalar path —
no rounding or precision is involved in a load, so output must match bit-for-bit.

If diffs change at all, that signals either:
- A predicate/coordinate mismatch (most likely cause: wrong `tXcX` partitioning).
- An OOB-row leak (most likely cause: missing `Clear_OOB_MN=true` or partition gap).

In either case, run with `DG_JIT_DEBUG=1` to print the launch dims and bisect by
reverting to scalar in one site at a time.
