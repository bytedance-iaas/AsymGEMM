# Merge Plan: Extend `cuda-graph-rewrite` to FP4 and Finalize

## Goal

The `cuda-graph-rewrite` working-tree changes already converted the FP8 masked and BF16
masked GEMM variants to the `masked_m`-direct API (constant `gridDim.y == num_groups`).
This plan extends the same pattern to the **FP4 masked** variant and then commits
everything as a clean merge of the CUDA-graph-safe rewrite into `main`.

---

## Context: What is Already Done

| File | Status |
|------|--------|
| `asym_gemm/include/asym_gemm/common/asymScheduler.cuh` | **Done** — new `if constexpr (MGroupedMasked)` branch reads `masked_m` directly |
| `asym_gemm/include/asym_gemm/impls/sm100_fp8_asym_gemm_1d1d.cuh` | **Done** — early-exit guard added |
| `asym_gemm/include/asym_gemm/impls/sm100_bf16_asym_gemm.cuh` | **Done** — early-exit guard added |
| `csrc/jit_kernels/impls/sm100_fp8_asym_gemm_1d1d.hpp` | **Done** — `masked_m` Args, grid = `num_groups` |
| `csrc/jit_kernels/impls/sm100_bf16_asym_gemm.hpp` | **Done** — same |
| `csrc/apis/gemm.hpp` (FP8 + BF16 masked) | **Done** — `(masked_m)` API, pybind11 updated |
| `tests/test_fp8.py`, `tests/test_bf16.py`, `tests/test_fp8_fp4.py` | **Done** — pass `masked_m` directly |

The FP4 masked variant (`SM100FP4AsymGemmMaskedRuntime`) still uses the old
`offsets`/`experts`/`list_size` triplet in all four layers below.

---

## Step-by-Step Changes Required

### Step 1 — `asym_gemm/include/asym_gemm/impls/sm100_fp4_asym_gemm_1d1d.cuh`

**Location**: immediately after the `asymScheduler` constructor call at line ~244–245,
before `const uint32_t num_total_k_blocks = ...`.

**Change**: add the empty-slot early-exit guard, identical to what was done in the FP8
and BF16 kernels.

```cpp
// Before (line ~244–246):
auto scheduler = asymScheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumMulticast, kIsMulticastOnA, kNumSMs>(
    shape_m, shape_n, experts, offsets);
const uint32_t num_total_k_blocks = ceil_div_device(shape_k, BLOCK_K);

// After:
auto scheduler = asymScheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumMulticast, kIsMulticastOnA, kNumSMs>(
    shape_m, shape_n, experts, offsets);

// Early-exit for inactive expert slots in the masked layout.
// With gridDim.y == num_groups (constant), slots whose token count is zero must
// return immediately. Safe here: cluster_sync() has completed and no TMA or TMEM
// barriers have been armed yet, so no barrier is left in an un-arrived state.
if constexpr (kGemmType == GemmType::MGroupedMasked) {
    if (scheduler.m_end == 0) return;
}

const uint32_t num_total_k_blocks = ceil_div_device(shape_k, BLOCK_K);
```

**Why safe**: same reasoning as FP8/BF16 — the guard is placed after `cluster_sync()`
(line ~241) and before any `full_barriers[i]->init()` or TMEM allocation, so no
barrier or tensor-memory allocation has been touched yet by this block.

---

### Step 2 — `csrc/jit_kernels/impls/sm100_fp4_asym_gemm_1d1d.hpp` (masked runtime only)

Only `SM100FP4AsymGemmMaskedRuntime` changes. The contiguous runtime
`SM100FP4AsymGemm1D1DRuntime` is **not touched**.

#### 2a. `Args` struct

```cpp
// Before (inside SM100FP4AsymGemmMaskedRuntime::Args):
void* offsets;
void* experts;

// After:
// masked_m: int32 device pointer [num_groups], token count per group.
// Passed to the kernel as the `offsets` argument; `experts` is unused (nullptr).
void* masked_m;
```

#### 2b. `launch_impl`

```cpp
// Before:
DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
    args.offsets, args.experts, args.m, args.n, args.k,
    args.tensor_map_a, args.tensor_map_b,
    args.tensor_map_sfa, args.tensor_map_sfb,
    args.tensor_map_cd));

// After:
DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
    args.masked_m, nullptr,   // offsets=masked_m, experts=nullptr (unused)
    args.m, args.n, args.k,
    args.tensor_map_a, args.tensor_map_b,
    args.tensor_map_sfa, args.tensor_map_sfb,
    args.tensor_map_cd));
```

#### 2c. `sm100_m_grouped_fp4_asym_gemm_masked_1d1d` function signature

```cpp
// Before:
static void sm100_m_grouped_fp4_asym_gemm_masked_1d1d(
    const torch::Tensor& a, const torch::Tensor& sfa,
    const torch::Tensor& b, const torch::Tensor& sfb,
    const torch::Tensor& d,
    const torch::Tensor& offsets_t,
    const torch::Tensor& experts_t,
    const int& list_size,
    const int& expected_m, ...)

// After:
static void sm100_m_grouped_fp4_asym_gemm_masked_1d1d(
    const torch::Tensor& a, const torch::Tensor& sfa,
    const torch::Tensor& b, const torch::Tensor& sfb,
    const torch::Tensor& d,
    const torch::Tensor& masked_m_t,
    const int& expected_m, ...)
```

#### 2d. Grid dimension and Args population

```cpp
// Before:
.launch_args = LaunchArgs({ceil_div(n, config.block_n), list_size - 1},
                          config.thread_config.num_threads,
                          config.smem_config.smem_size,
                          config.multicast_config.num_multicast),
.offsets = offsets_t.data_ptr<int>(),
.experts = experts_t.data_ptr<int>(),

// After — gridDim.y == num_groups (constant); CUDA-graph safe:
.launch_args = LaunchArgs({ceil_div(n, config.block_n), num_groups},
                          config.thread_config.num_threads,
                          config.smem_config.smem_size,
                          config.multicast_config.num_multicast),
.masked_m = masked_m_t.data_ptr<int>(),
```

---

### Step 3 — `csrc/apis/gemm.hpp`

#### 3a. `m_grouped_fp4_asym_gemm_nt_masked` function signature and validation

```cpp
// Before:
static void m_grouped_fp4_asym_gemm_nt_masked(
    const std::pair<torch::Tensor, torch::Tensor>& a,
    const std::pair<torch::Tensor, torch::Tensor>& b,
    const torch::Tensor& d,
    const torch::Tensor& offsets_t,
    const torch::Tensor& experts_t,
    const int& list_size,
    const int& expected_m, ...)

// After:
static void m_grouped_fp4_asym_gemm_nt_masked(
    const std::pair<torch::Tensor, torch::Tensor>& a,
    const std::pair<torch::Tensor, torch::Tensor>& b,
    const torch::Tensor& d,
    const torch::Tensor& masked_m,
    const int& expected_m, ...)
```

Add validation (the current FP4 function has no offsets/experts assertions; add the
`masked_m` assertions in the same place that the FP8 function has them, after the
scalar-type checks):

```cpp
// Add after existing shape/dtype assertions:
// masked_m: int32 GPU tensor of shape [num_groups] with per-group token counts
DG_HOST_ASSERT(masked_m.is_cuda() && masked_m.is_contiguous());
DG_HOST_ASSERT(masked_m.scalar_type() == torch::kInt);
DG_HOST_ASSERT(masked_m.numel() == num_groups);
```

#### 3b. Dispatch call

```cpp
// Before:
sm100_m_grouped_fp4_asym_gemm_masked_1d1d(a.first, sfa, b.first, sfb, d,
    offsets_t, experts_t, list_size, expected_m,
    num_groups, m, n, k, major_a, major_b, compiled_dims);

// After:
sm100_m_grouped_fp4_asym_gemm_masked_1d1d(a.first, sfa, b.first, sfb, d,
    masked_m, expected_m,
    num_groups, m, n, k, major_a, major_b, compiled_dims);
```

#### 3c. pybind11 registration

```cpp
// Before:
m.def("m_grouped_fp4_asym_gemm_nt_masked",
    static_cast<void(*)(const std::pair<torch::Tensor, torch::Tensor>&,
                        const std::pair<torch::Tensor, torch::Tensor>&,
                        const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const int&, const int&,
                        std::optional<std::tuple<int, int, int>>, const std::string&, const bool&)>(
        &m_grouped_fp4_asym_gemm_nt_masked),
    py::arg("a"), py::arg("b"), py::arg("d"),
    py::arg("offsets"), py::arg("experts"), py::arg("list_size"), py::arg("expected_m"),
    py::arg("recipe") = std::nullopt, py::arg("compiled_dims") = "nk",
    py::arg("disable_ue8m0_cast") = false);

// After:
m.def("m_grouped_fp4_asym_gemm_nt_masked",
    static_cast<void(*)(const std::pair<torch::Tensor, torch::Tensor>&,
                        const std::pair<torch::Tensor, torch::Tensor>&,
                        const torch::Tensor&, const torch::Tensor&, const int&,
                        std::optional<std::tuple<int, int, int>>, const std::string&, const bool&)>(
        &m_grouped_fp4_asym_gemm_nt_masked),
    py::arg("a"), py::arg("b"), py::arg("d"),
    py::arg("masked_m"), py::arg("expected_m"),
    py::arg("recipe") = std::nullopt, py::arg("compiled_dims") = "nk",
    py::arg("disable_ue8m0_cast") = false);
```

---

### Step 4 — `tests/test_nvfp4.py`

#### 4a. Remove the CPU-side helper

Delete the `_build_offsets_experts_from_masked_m` function (lines ~438–466) entirely.
It performs `.item()` calls inside a Python loop and is replaced by the direct
`masked_m` API.

#### 4b. Remove `import ipdb; ipdb.set_trace()` debug line

This debug breakpoint (line ~499) was left in the test. Remove it.

#### 4c. Update the call site in `test_m_grouped_nvfp4_masked_cpp_flow`

```python
# Before:
offsets_t, experts_t, list_size = _build_offsets_experts_from_masked_m(
    masked_m_cpu, num_groups, max_m, block_m=block_m
)
# ... (masked_m already built as masked_m = masked_m_cpu.to(device="cuda"))
asym_gemm.m_grouped_fp4_asym_gemm_nt_masked(
    (a_fp4, sfa),
    (b_fp4, sfb),
    d_kernel,
    offsets_t,
    experts_t,
    list_size,
    expected_m_per_group,
    recipe=recipe,
    disable_ue8m0_cast=disable_ue8m0_cast,
)

# After:
# (remove offsets_t/experts_t/list_size construction)
asym_gemm.m_grouped_fp4_asym_gemm_nt_masked(
    (a_fp4, sfa),
    (b_fp4, sfb),
    d_kernel,
    masked_m,
    expected_m_per_group,
    recipe=recipe,
    disable_ue8m0_cast=disable_ue8m0_cast,
)
```

The `masked_m` tensor (already on CUDA) is passed directly; no CPU-side loop or
D2H sync.

---

## What Does NOT Change

| File / Item | Reason |
|-------------|--------|
| `SM100FP4AsymGemm1D1DRuntime` (contiguous) in `.hpp` | Contiguous variant keeps `offsets`/`experts`; this path is not being CUDA-graph-ified in this PR |
| `sm100_m_grouped_fp4_asym_gemm_contiguous_1d1d` in `.hpp` | Same |
| `m_grouped_fp4_asym_gemm_nt_contiguous` in `gemm.hpp` | Same |
| `asymScheduler.cuh` | Already done; the `else` branch covers contiguous correctly |
| `tests/test_fp8_fp4.py` (FP4 contiguous path) | Uses `offsets`/`experts` for contiguous — untouched |

---

## Commit Strategy

All changes above are already present in the working tree for FP8/BF16. After completing
Steps 1–4 for FP4, the full set of modified files should be committed together as a
single atomic commit on `main`:

```
Files to stage:
  asym_gemm/include/asym_gemm/common/asymScheduler.cuh
  asym_gemm/include/asym_gemm/impls/sm100_fp8_asym_gemm_1d1d.cuh
  asym_gemm/include/asym_gemm/impls/sm100_bf16_asym_gemm.cuh
  asym_gemm/include/asym_gemm/impls/sm100_fp4_asym_gemm_1d1d.cuh   ← Step 1
  csrc/jit_kernels/impls/sm100_fp8_asym_gemm_1d1d.hpp
  csrc/jit_kernels/impls/sm100_bf16_asym_gemm.hpp
  csrc/jit_kernels/impls/sm100_fp4_asym_gemm_1d1d.hpp               ← Step 2
  csrc/apis/gemm.hpp                                                 ← Step 3
  tests/test_fp8.py
  tests/test_bf16.py
  tests/test_fp8_fp4.py
  tests/test_nvfp4.py                                                ← Step 4
```

Suggested commit message:
```
cuda-graph-rewrite: replace offsets/experts/list_size with masked_m for all masked GEMMs

Replace the variable-length offsets/experts/list_size triplet with a flat
masked_m[num_groups] int32 tensor for FP8, BF16, and FP4 masked GEMM variants.

Key changes:
- asymScheduler: new if constexpr(MGroupedMasked) branch reads masked_m
  directly; gridDim.y == num_groups (compile-time constant)
- All three masked kernels: early-exit guard for zero-token slots, placed
  after cluster_sync() and before any TMA/TMEM barrier init
- JIT runtimes: Args struct and launch_impl updated; grid Y = num_groups
- C++ API: function signatures and pybind11 registrations simplified
- Tests: remove CPU-side build_offsets_experts_from_masked_m helpers

Result: masked GEMM launches are now CUDA-graph capturable — gridDim.y
and buffer shapes are model constants independent of routing.
```

---

## Verification Checklist

After making the changes, verify:

- [ ] `tests/test_fp8.py::test_m_grouped_gemm_masked` passes — FP8 masked correctness
- [ ] `tests/test_bf16.py::test_m_grouped_gemm_masked` passes — BF16 masked correctness
- [ ] `tests/test_fp8_fp4.py::test_m_grouped_gemm_masked` passes — FP8 masked via fp4 test
- [ ] `tests/test_nvfp4.py::test_m_grouped_nvfp4_masked_cpp_flow` passes — FP4 masked correctness
- [ ] Contiguous path tests still pass (no regression from the untouched code paths)
- [ ] Confirm `gridDim.y == num_groups` is constant by printing grid dims in a debug run
