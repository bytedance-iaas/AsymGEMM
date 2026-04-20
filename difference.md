# Differences: `cuda-graph-rewrite` vs `main`

> **Note**: The `cuda-graph-rewrite` branch does not exist as a remote branch.
> The differences below reflect the **current working-tree modifications** (unstaged
> relative to `HEAD`/`main`) which implement the CUDA-graph compatibility changes
> described in `cudaGraph.md`.

---

## Overview

The rewrite eliminates the Python-level `offsets`/`experts`/`list_size` triplet for the
**masked** GEMM variant in favour of passing `masked_m` (a flat `int32[num_groups]` token-count
array) directly.  This makes grid dimensions a compile-time constant (`num_groups`) so the
masked kernels become CUDA-graph safe.

---

## 1. `asym_gemm/include/asym_gemm/common/asymScheduler.cuh`

### What changed
The `MGroupedMasked` initialisation path in the `asymScheduler` constructor was rewritten.
Previously every variant shared a single code path that indexed into `offsets[]` pairs and
`experts[]`.  After the change, `MGroupedMasked` gets its own `if constexpr` branch.

### Before (single shared path)
```cpp
expert_id = experts[blockIdx.y];
n_start    = expert_id * blocks_perExpert;

uint32_t offset_pair_idx = blockIdx.y * 2;
m_start = ceil_div_device(offsets[offset_pair_idx],     BLOCK_M);
m_end   = ceil_div_device(offsets[offset_pair_idx + 1], BLOCK_M);

n_idx = blockIdx.x * BLOCK_N + shape_n * expert_id;
current_group_idx = expert_id;
```

### After (split by GemmType)
```cpp
if constexpr (kGemmType == GemmType::MGroupedMasked) {
    // `offsets` is now reinterpreted as int32 masked_m[num_groups].
    // `experts` is unused.  blockIdx.y IS the expert id (gridDim.y == num_groups).
    expert_id         = blockIdx.y;
    current_group_idx = blockIdx.y;
    n_idx             = blockIdx.x * BLOCK_N + shape_n * expert_id;
    const int m_count = reinterpret_cast<const int*>(offsets)[blockIdx.y];
    m_start = 0;
    m_end   = ceil_div_device(static_cast<uint32_t>(m_count > 0 ? m_count : 0), BLOCK_M);
} else {
    // ... original pair-offsets path (unchanged) ...
}
```

### Key semantic differences
| Aspect | `main` | `cuda-graph-rewrite` |
|--------|--------|----------------------|
| `offsets` pointer | Pairs `[start_0,end_0, start_1,end_1,…]` of length `2*list_size` | Flat `masked_m[num_groups]` token counts |
| `experts` pointer | `experts[blockIdx.y]` used to look up group id | Unused (`nullptr` passed from host) |
| `blockIdx.y` meaning | Index into compact active-expert list | Directly the expert/group id |
| `m_start` | `ceil_div(offsets[2*i],   BLOCK_M)` (non-zero for masked layout) | Always `0` |
| `m_end`   | `ceil_div(offsets[2*i+1], BLOCK_M)` | `ceil_div(masked_m[blockIdx.y], BLOCK_M)` |
| Grid Y dimension | `list_size - 1` (variable, routing-dependent) | `num_groups` (constant) |

---

## 2. `asym_gemm/include/asym_gemm/impls/sm100_fp8_asym_gemm_1d1d.cuh`

### What changed
An **early-exit guard** was inserted immediately after scheduler construction for the
`MGroupedMasked` case.

```cpp
// NEW — added after scheduler construction, before pipeline setup:
if constexpr (kGemmType == GemmType::MGroupedMasked) {
    if (scheduler.m_end == 0) return;
}
```

### Why it is needed
With `gridDim.y == num_groups` (constant), thread blocks are launched for every expert
slot including inactive ones (token count == 0).  Without the guard these blocks would
proceed into the TMA/TMEM pipeline and either compute garbage or stall on barriers
that no producer block will ever satisfy.

### Why it is safe
The guard sits after `cluster_sync()` (all cluster barriers resolved) and before any TMA
descriptor or TMEM barrier is initialised, so no un-arrived barrier can be left behind.

---

## 3. `asym_gemm/include/asym_gemm/impls/sm100_bf16_asym_gemm.cuh`

Identical early-exit guard added in the same position relative to scheduler construction:

```cpp
if constexpr (kGemmType == GemmType::MGroupedMasked) {
    if (scheduler.m_end == 0) return;
}
```

No other changes to the BF16 kernel body.

---

## 4. `csrc/jit_kernels/impls/sm100_fp8_asym_gemm_1d1d.hpp`

### Args struct
```cpp
// Before
void* offsets;
void* experts;

// After
// masked_m: int32 device pointer [num_groups], token count per group.
// Passed to the kernel as the `offsets` argument; `experts` is unused (nullptr).
void* masked_m;
```

### `launch_impl`
```cpp
// Before
args.offsets, args.experts, args.m, args.n, args.k, …

// After
args.masked_m, nullptr,   // offsets=masked_m, experts=nullptr (unused)
args.m, args.n, args.k, …
```

### `sm100_m_grouped_fp8_asym_gemm_masked_1d1d` function signature
```cpp
// Before
const torch::Tensor& offsets_t,
const torch::Tensor& experts_t,
const int& list_size,

// After
const torch::Tensor& masked_m_t,
```

### Grid dimension
```cpp
// Before
.launch_args = LaunchArgs({ceil_div(n, config.block_n), list_size - 1}, …)

// After — gridDim.y is now a compile-time constant; CUDA-graph safe
.launch_args = LaunchArgs({ceil_div(n, config.block_n), num_groups}, …)
```

### Args population
```cpp
// Before
.offsets = offsets_t.data_ptr<int>(),
.experts = experts_t.data_ptr<int>(),

// After
.masked_m = masked_m_t.data_ptr<int>(),
```

---

## 5. `csrc/jit_kernels/impls/sm100_bf16_asym_gemm.hpp`

All changes are identical to §4 (same structural rewrite for the BF16 masked variant):

- `Args` struct: `offsets`/`experts` → `masked_m`
- `launch_impl`: passes `(masked_m, nullptr, …)` instead of `(offsets, experts, …)`
- Function signature: `(offsets_t, experts_t, list_size)` → `(masked_m_t)`
- Grid: `list_size - 1` → `num_groups`

---

## 6. `csrc/apis/gemm.hpp`

### `m_grouped_fp8_asym_gemm_nt_masked`

**Function signature**:
```cpp
// Before
const torch::Tensor& offsets_t,
const torch::Tensor& experts_t,
const int& list_size,

// After
const torch::Tensor& masked_m,
```

**Validation**:
```cpp
// Before
DG_HOST_ASSERT(offsets_t.is_cuda() && experts_t.is_cuda());
DG_HOST_ASSERT(offsets_t.is_contiguous() && experts_t.is_contiguous());
DG_HOST_ASSERT(offsets_t.scalar_type() == torch::kInt && experts_t.scalar_type() == torch::kInt);
DG_HOST_ASSERT(offsets_t.numel() >= list_size && experts_t.numel() >= list_size);

// After — simpler: just validate masked_m shape
DG_HOST_ASSERT(masked_m.is_cuda() && masked_m.is_contiguous());
DG_HOST_ASSERT(masked_m.scalar_type() == torch::kInt);
DG_HOST_ASSERT(masked_m.numel() == num_groups);
```

**Dispatch call**:
```cpp
// Before
sm100_m_grouped_fp8_asym_gemm_masked_1d1d(…, offsets_t, experts_t, list_size, expected_m, …)

// After
sm100_m_grouped_fp8_asym_gemm_masked_1d1d(…, masked_m, expected_m, …)
```

### `m_grouped_bf16_asym_gemm_nt_masked`

Same API simplification: `(offsets, experts, list_size)` → `(masked_m)`.

### pybind11 registrations

```cpp
// Before — 8 positional args for fp8_masked
py::arg("a"), py::arg("b"), py::arg("d"),
py::arg("offsets"), py::arg("experts"), py::arg("list_size"), py::arg("expected_m"), …

// After — 6 positional args
py::arg("a"), py::arg("b"), py::arg("d"),
py::arg("masked_m"), py::arg("expected_m"), …
```

BF16 masked binding similarly simplified.

---

## 7. Test Files

### `tests/test_fp8.py`

```python
# Removed (both in correctness and benchmark sections):
offsets, experts, list_size = build_offsets_experts_from_masked_m(masked_m, num_groups, max_m)

# Before
asym_gemm.m_grouped_fp8_asym_gemm_nt_masked(a, b, d_asym, offsets, experts, list_size, expected_m_per_group, …)

# After — masked_m passed directly
asym_gemm.m_grouped_fp8_asym_gemm_nt_masked(a, b, d_asym, masked_m, expected_m_per_group, …)
```

### `tests/test_bf16.py`

```python
# Removed:
offsets, experts, list_size = build_offsets_experts_from_masked_m(masked_m, num_groups, max_m)

# Before
asym_gemm.m_grouped_bf16_asym_gemm_nt_masked(a, b_pinned, d_asym, offsets, experts, list_size, expected_m_per_group, …)

# After
asym_gemm.m_grouped_bf16_asym_gemm_nt_masked(a, b_pinned, d_asym, masked_m, expected_m_per_group, …)
```

### `tests/test_fp8_fp4.py`

```python
# Removed:
offsets, experts, list_size = build_offsets_experts_from_masked_m(masked_m, num_groups, max_m)

# Before
asym_gemm.m_grouped_fp8_asym_gemm_nt_masked(a, b, d_asym, offsets, experts, list_size, expected_m_per_group, …)

# After
asym_gemm.m_grouped_fp8_asym_gemm_nt_masked(a, b, d_asym, masked_m, expected_m_per_group, …)
```

---

## Summary Table

| File | Nature of change |
|------|-----------------|
| `asymScheduler.cuh` | New `if constexpr (MGroupedMasked)` branch reinterprets `offsets` as flat `masked_m`; `experts` becomes unused; `m_start` always 0; `blockIdx.y` == expert id directly |
| `sm100_fp8_asym_gemm_1d1d.cuh` | Early-exit guard `if (scheduler.m_end == 0) return` for inactive slots |
| `sm100_bf16_asym_gemm.cuh` | Same early-exit guard |
| `sm100_fp8_asym_gemm_1d1d.hpp` | `Args`: `offsets`/`experts` → `masked_m`; grid Y: `list_size-1` → `num_groups`; launch passes `(masked_m, nullptr)` |
| `sm100_bf16_asym_gemm.hpp` | Same as above for BF16 |
| `csrc/apis/gemm.hpp` | Both masked API signatures simplified to `(masked_m)`; pybind11 registrations updated |
| `tests/test_fp8.py` | Remove `build_offsets_experts_from_masked_m`; pass `masked_m` directly |
| `tests/test_bf16.py` | Same |
| `tests/test_fp8_fp4.py` | Same |

## CUDA-Graph Safety Gain

| Property | `main` | `cuda-graph-rewrite` |
|----------|--------|----------------------|
| `gridDim.y` | `list_size - 1` (variable per step) | `num_groups` (model constant) |
| Buffer shapes | `offsets[2*active]`, `experts[active+1]` (variable) | `masked_m[num_groups]` (fixed) |
| Host↔device sync | `.item()` / `.tolist()` in `build_offsets_experts_from_masked_m` | None — kernel reads `masked_m` from GPU |
| CUDA graph capturable | No | Yes |
