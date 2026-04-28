# FP8 Masked GEMM: `masked_m > expected_m_per_group` and Kernel Warmup

---

## 1. Is `masked_m[j] > expected_m_per_group` Correct?

**Yes, it is by design and is expected behaviour.**

### Where the values come from

`generate_m_grouped_masked` (generators.py line 334–336):

```python
for j in range(num_groups):
    masked_m[j] = int(expected_m_per_group * random.uniform(0.7, 1.3))
```

Each group's token count is drawn uniformly from `[0.7 × expected, 1.3 × expected]`.  
With `expected_m_per_group = 1024` the range is `[716, 1331]`, so values such as `1198` or `1274`
appear naturally. The only hard constraint (line 337) is:

```python
assert masked_m.amax().item() <= max_m   # max_m = 4096, not expected_m_per_group
```

The allocation `a[G, max_m, K]` is sized to `max_m` (here 4096), so all observed counts fit safely.

### What `expected_m_per_group` means to the kernel

`expected_m_per_group` is a **compile-time / grid-dimension hint**, not an upper bound:

| Role | Description |
|---|---|
| Grid selection | Determines how many CTAs are launched along the M axis (`ceil(expected / BLOCK_M)`) |
| Tile heuristics | Influences BLOCK_M, number of pipeline stages, and wave efficiency choices |
| Correctness | Irrelevant — the scheduler iterates from `m_start` to `ceil(masked_m[g] / BLOCK_M)` regardless |

When `masked_m[g] > expected_m_per_group` the kernel is **correct** but some M-blocks that
would have been processed by separate CTAs are instead serialised within a single CTA.  
This is the source of the existing TODO comment at `test_fp8.py:338`:

```python
# TODO: when the actual `m` is greater than `expected_m_per_group`, efficiency may significantly decrease.
```

### Summary

| Condition | Correctness | Performance |
|---|---|---|
| `masked_m[g] < expected_m_per_group` | ✓ | Optimal — grid is right-sized |
| `masked_m[g] == expected_m_per_group` | ✓ | Optimal |
| `masked_m[g] > expected_m_per_group` | ✓ | Degraded — extra M-blocks are serialised |
| `masked_m[g] > max_m` | ✗ | Out-of-bounds memory access |

The test exercises the `> expected` case intentionally (the ±30 % range) to confirm correctness
under real-world token imbalance.

---

## 2. Warmup Pattern — Making FP8 Match FP4

### The FP4 warmup pattern (in `test_nvfp4.py`)

```python
# expected_m_per_group = 1
# masked_m = torch.zeros(
#     (num_groups,), dtype=torch.int32, device="cuda"
# )
```

The intent is to fire the kernel once with zero work to trigger JIT compilation before profiling.

### Problem with `expected_m_per_group = 1`

The kernel is JIT-compiled for a specific `expected_m_per_group` (it affects the grid dimensions
and, on SM100, some compile-time constants selected by the heuristic).  
Warming up with `expected_m_per_group = 1` compiles a **different variant** than the one used at
`expected_m_per_group = 1024`. The first production call still incurs the JIT latency.

The correct approach is: keep `expected_m_per_group` at its production value but pass
`masked_m = zeros` so every group has `m_end = 0` and the kernel returns immediately after the
early-exit check:

```cpp
// sm100_fp8_asym_gemm_1d1d.cuh (and fp4 variant)
if constexpr (kGemmType == GemmType::MGroupedMasked) {
    if (scheduler.m_end == 0) return;
}
```

### Where to add the warmup in `test_fp8.py`

The pre-transformed tensors `a_bench` / `b_bench` already exist at the profiling site
(lines 410–415). Insert a warmup call immediately before `bench_kineto`:

**File**: `tests/test_fp8.py`, between lines 415 and 422 (after `b_bench` is built, before `test_func`):

```python
# Warmup: compile the kernel for this (expected_m_per_group, n, k) without
# doing any computation. masked_m = zeros makes every group exit immediately.
_warmup_masked_m = torch.zeros((num_groups,), device='cuda', dtype=torch.int32)
asym_gemm.m_grouped_fp8_asym_gemm_nt_masked(
    a_bench, b_bench, d_asym, _warmup_masked_m, expected_m_per_group,
    disable_ue8m0_cast=disable_ue8m0_cast
)
torch.cuda.synchronize()
```

Place this **after** `b_bench` is defined (line 415) and **before** `test_func` / `test_func_asym`
are called (line 418).

### Also update the FP4 warmup comment

In `test_nvfp4.py`, the commented-out warmup should be changed from `expected_m_per_group = 1`
to `masked_m = zeros` with the real `expected_m_per_group`:

```python
# Correct warmup (replace the current commented-out block):
_warmup_masked_m = torch.zeros((num_groups,), device='cuda', dtype=torch.int32)
asym_gemm.m_grouped_fp4_asym_gemm_nt_masked(
    (a_fp4, sfa), (b_fp4, sfb), d_kernel,
    _warmup_masked_m, expected_m_per_group,
    recipe=recipe, disable_ue8m0_cast=disable_ue8m0_cast,
)
torch.cuda.synchronize()
```

---

## 3. Complete Change Summary

### `tests/test_fp8.py`

Insert after line 415 (`b_bench = (b[0], sfb_pre)`):

```python
# Warmup: trigger JIT compilation for this (expected_m_per_group, n, k) shape
# without doing any GEMM work (all groups have 0 tokens → early exit in kernel).
_warmup_masked_m = torch.zeros((num_groups,), device='cuda', dtype=torch.int32)
asym_gemm.m_grouped_fp8_asym_gemm_nt_masked(
    a_bench, b_bench, d_asym, _warmup_masked_m, expected_m_per_group,
    disable_ue8m0_cast=disable_ue8m0_cast
)
torch.cuda.synchronize()
```

No other changes to the test logic are needed.

### `tests/test_nvfp4.py`

Replace the commented-out warmup block:
```python
# Before (incorrect: compiles wrong variant)
# expected_m_per_group = 1
# masked_m = torch.zeros(
#     (num_groups,), dtype=torch.int32, device="cuda"
# )

# After (correct: compiles production variant, zero work)
# _warmup_masked_m = torch.zeros((num_groups,), dtype=torch.int32, device="cuda")
# asym_gemm.m_grouped_fp4_asym_gemm_nt_masked(
#     (a_fp4, sfa), (b_fp4, sfb), d_kernel,
#     _warmup_masked_m, expected_m_per_group,
#     recipe=recipe, disable_ue8m0_cast=disable_ue8m0_cast,
# )
# torch.cuda.synchronize()
```

---

## 4. Why `masked_m = zeros` Is Safe as a Warmup

The kernel grid is `gridDim = (n_blocks, num_groups)`. Every CTA checks:

```cpp
if (scheduler.m_end == 0) return;
```

`scheduler.m_end` is derived from `masked_m[blockIdx.y]`. When `masked_m[g] = 0` for all `g`,
every CTA exits immediately after the cluster sync. No TMA loads, no UMMA instructions, no
epilogue writes fire — but the kernel *launches*, which is enough to:

1. Trigger the JIT compiler (first call only)
2. Warm up L2 cache residency of the kernel binary
3. Populate any CUDA context state needed for subsequent calls

The warmup adds negligible overhead (< 10 µs for the grid launch + immediate exits).
