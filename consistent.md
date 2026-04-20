# Consistency Plan: FP4 Decode Test, Offset Semantics, Non-Asym Removal, FP4 BLOCK_M

This document is the plan for four consistency follow-ups. No code changes are
made here — implementation happens after review.

---

## 1. FP4 Masked (Decode) Phase Test

### Goal
Match the FP8 coverage pattern: today `tests/test_fp8.py` (via `test_fp8_fp4.py`)
exercises both prefill (contiguous) and decode (masked) phases, but
`tests/test_nvfp4.py` only tests prefill. Add a masked-phase test for FP4
that exercises `m_grouped_fp4_asym_gemm_nt_masked`.

### FP8 decode pattern to mirror
`tests/test_fp8_fp4.py`:
- `generate_m_grouped_masked(num_groups, max_m, expected_m_per_group, n, k, ...)`
  returns `(a, b, masked_m, psum_m, d, ref_d)` with `a` shaped `(G, max_m, K)`.
- `build_offsets_experts_from_masked_m(masked_m, num_groups, max_m, block_m=128)`
  builds the pairs-format offsets:
  - `start = g * max_m`
  - `end   = start + ceil_div(masked_m[g], block_m) * block_m`
  - skip groups with `masked_m[g] == 0`
  - append experts list terminated with `-1`
- Kernel is invoked as
  `m_grouped_fp8_asym_gemm_nt_masked(a, b, d, offsets_t, experts_t, list_size, expected_m, recipe, ...)`.

### FP4 changes vs FP8
- Use the FP4 quant path already supported by `generate_m_grouped_masked`
  (`quant_config.is_fp4_a/is_fp4_b`), i.e. E2M1 packed activations + E4M3 SF.
- Call `m_grouped_fp4_asym_gemm_nt_masked` (already exported in
  `asym_gemm/__init__.py`).
- `BLOCK_M` for the offset helper stays `128` (see §4).
- Expected output shape matches FP8 masked: `(G, max_m, N)` in BF16/FP32.

### Test location
Add a new function `test_m_grouped_nvfp4_masked_cpp_flow` to
`tests/test_nvfp4.py` (alongside the existing contiguous test). Numerical
tolerance should match the contiguous FP4 test (e.g. mean_abs ~0.05,
max_abs ~1.0, rel ~0.005); use the same reference path
(`ref_d` returned by `generate_m_grouped_masked`).

### Acceptance
- `python tests/test_nvfp4.py` runs both contiguous and masked tests.
- Masked test passes tolerance and does not regress contiguous path.

---

## 2. Offset `[start, end]` Convention: `n*BLOCK_M` or `n*BLOCK_M - 1`?

### Answer
**Use `n * BLOCK_M` (exclusive upper bound, aligned up).** Both values lead
to the same scheduled tile count, but the exclusive convention is already
what the codebase assumes and is less error-prone at boundaries.

### Evidence from the kernels
`asym_gemm/include/asym_gemm/common/asymScheduler.cuh` (masked branch):

```cpp
uint32_t offset_pair_idx = blockIdx.y * 2;
m_start = ceil_div_device(offsets[offset_pair_idx],     BLOCK_M);
m_end   = ceil_div_device(offsets[offset_pair_idx + 1], BLOCK_M);
```

- `ceil_div(n * BLOCK_M,     BLOCK_M) = n`
- `ceil_div(n * BLOCK_M - 1, BLOCK_M) = n`   (only if `n ≥ 1`)
- `ceil_div(0, BLOCK_M)                 = 0` (safe for empty expert)

Both FP8 (`sm100_fp8_asym_gemm_1d1d.cuh`) and FP4
(`sm100_fp4_asym_gemm_1d1d.cuh`) consume the same scheduler, so the
convention is identical.

### Edge cases that disqualify `n*BLOCK_M - 1`
- `masked_m[g] == 0`: `n = 0`, so `end` must be `start` (equal to `start`),
  not `start - 1`. Empty experts are filtered out by
  `build_offsets_experts_from_masked_m` today; using `n*BLOCK_M` keeps the
  invariant `end > start` for listed experts and `end == start` would be
  trivially skippable.
- Using inclusive-minus-one forces every caller to special-case `n == 0`,
  while `ceil_div` on the exclusive bound handles it naturally.

### Action
Keep the helper in `tests/test_fp8_fp4.py` as-is (`n*BLOCK_M`), and use the
same formula in the new FP4 masked test. No code change to the scheduler.

---

## 3. Remove Non-Asym Kernels and Their Definitions

Goal: since this project is asym-only, drop legacy non-asym GEMM
entrypoints and headers so the API surface and includes are unambiguous.

### Files to delete
- `csrc/jit_kernels/impls/sm100_fp8_gemm_1d1d.hpp`
  (defines `sm100_m_grouped_fp8_gemm_contiguous_1d1d`; only caller is the
  non-asym wrapper we are removing).
- `csrc/jit_kernels/impls/sm100_bf16_gemm.hpp`
  (defines `sm100_m_grouped_bf16_gemm_contiguous`; only caller is the
  non-asym wrapper we are removing).
- `asym_gemm/include/asym_gemm/impls/sm100_fp8_gemm_1d1d.cuh`
  (only used by the above .hpp).
- `asym_gemm/include/asym_gemm/impls/sm100_bf16_gemm.cuh`
  (only used by the above .hpp).

Before deleting, grep for each symbol to confirm there are no other
consumers. (Known-safe: asym kernels live in
`sm100_fp8_asym_gemm_1d1d.{cuh,hpp}` and `sm100_bf16_asym_gemm.{cuh,hpp}`
and do NOT include the non-asym headers.)

### `csrc/apis/gemm.hpp`
Remove:
- line 10: `#include "../jit_kernels/impls/sm100_bf16_gemm.hpp"`
- line 12: `#include "../jit_kernels/impls/sm100_fp8_gemm_1d1d.hpp"`
- lines 113–158: `m_grouped_fp8_gemm_nt_contiguous(...)`
- lines 425–457: `m_grouped_bf16_gemm_nt_contiguous(...)`
- in `register_apis`:
  - line 513: `m.def("m_grouped_fp8_gemm_nt_contiguous", ...)`
  - line 560: `m.def("m_grouped_bf16_gemm_nt_contiguous", ...)`

Keep the asym versions (`m_grouped_fp8_asym_gemm_*`,
`m_grouped_bf16_asym_gemm_*`, `m_grouped_fp4_asym_gemm_*`) untouched.

### `csrc/indexing/main.cu`
Remove lines 5–6:
```cpp
#include <asym_gemm/impls/sm100_bf16_gemm.cuh>
#include <asym_gemm/impls/sm100_fp8_gemm_1d1d.cuh>
```
(Confirm nothing in `indexing/main.cu` instantiates the removed templates.)

### `csrc/apis/einsum.hpp` (dead code)
`python_api.cpp` does not include `einsum.hpp`, so nothing is bound from it
today. Still, to avoid future confusion, remove its stale references:
- `#include "../jit_kernels/impls/sm90_bf16_gemm.hpp"`
- `#include "../jit_kernels/impls/sm100_bf16_gemm.hpp"`
and any `sm100_bf16_bhr_hdr_bhd`/`sm90_bf16_bhr_hdr_bhd` references that
do not resolve. (If the whole file is dead, flag it in the PR description
but do not delete unless confirmed with the user.)

### `asym_gemm/__init__.py`
Remove the stale import name (line 56):
```python
"m_grouped_bf16_gemm_nt_contiguous",
```
Note: `"m_grouped_fp8_gemm_nt_contiguous"` is not currently listed, so no
change needed there. Also drop legacy alias exports if they target
now-removed symbols (none do — current aliases point at the asym masked
kernels, which remain).

### Python tests / callers
Grep `tests/` and top-level scripts for `m_grouped_fp8_gemm_nt_contiguous`
and `m_grouped_bf16_gemm_nt_contiguous`; update or remove any callers
before dropping the bindings. (Expected: none in the active test suite.)

### Build verification
After edits:
1. `bash install.sh` must succeed.
2. `python tests/test_fp8.py`, `python tests/test_fp8_fp4.py`,
   `python tests/test_nvfp4.py` must all still pass.

---

## 4. FP4 `BLOCK_M`: 128 or 256?

### Answer
**`BLOCK_M = 128` (same as FP8).** Only `BLOCK_K` changes for FP4.

### Why
`BLOCK_M` counts **rows** of A, not bytes. The NVFP4 packing (two 4-bit
values per byte) only compresses along the **K** dimension. Concretely,
in `asym_gemm/include/asym_gemm/impls/sm100_fp4_asym_gemm_1d1d.cuh`:

```cpp
SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K / 2;   // /2 is K-axis packing
DG_STATIC_ASSERT(BLOCK_K % 64 == 0, "block K must be a multiple of 64 for FP4");
// TMA uses byte coords on the K axis:
const uint32_t k_packed_idx = k_block_idx * (BLOCK_K / 2);
```

And in `csrc/jit_kernels/impls/sm100_fp4_asym_gemm_1d1d.hpp` (both the
contiguous path at line ~109 and the masked path at line ~304):

```cpp
const int block_m = 128;    // rows
const int block_k = 512;    // 2x FP8's 256
```

So the per-row SMEM byte footprint for a K-block stays the same as FP8
(256 bytes per row per k-block), while the effective arithmetic K-tile
doubles. The MMA tile shape (and therefore the natural row tile) does not
change when going from FP8 → FP4.

### Action
No change. Keep `block_m = 128` in the FP4 masked test’s offset helper so
it matches the kernel’s scheduler tile.

---

## Implementation Order (after approval)

1. Write FP4 masked test in `tests/test_nvfp4.py` (no kernel changes).
2. Run `bash install.sh` and confirm new test passes.
3. Remove non-asym entrypoints, bindings, headers, and includes per §3.
4. Rebuild and re-run `test_fp8.py`, `test_fp8_fp4.py`, `test_nvfp4.py`.
5. Commit in two logical chunks: (a) FP4 masked test, (b) non-asym removal.
