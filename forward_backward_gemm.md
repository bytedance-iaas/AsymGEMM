# Forward/Backward AsymGEMM: Single-Tensor Transposed TMA Descriptor

## Problem Statement

Today `AsymGEMMMoEFunction` carries **two** CPU-pinned tensors per weight matrix:

```
gate_up      [E, 2I, H]  K-major (stride(-1)=1)   → forward GEMM
gate_up_T    [E,  H, 2I] non-contiguous view       → backward GEMM (Phase 4)
```

The view `gate_up.permute(0,2,1)` costs nothing in DRAM (same storage), but it routes the layout decision through Python's stride-detection path and couples the kernel's behaviour to Python object semantics.

The proposal: pass **one tensor** (`gate_up [E, 2I, H]`) for both directions and tell the kernel explicitly which TMA layout to build. No Python permutation, no implicit stride inspection.

---

## Background: How AsymGEMM Selects a TMA Descriptor

The chain, from Python call to tensor-core descriptor:

```
Python tensor strides
        │
        ▼
get_major_type_ab()          layout.hpp:21
  stride(-1)==1 → Major::K
  stride(-1)!=1 → Major::MN
        │
        ▼
make_tma_b_desc()            runtime_utils.hpp:168
  get_inner_outer_dims(major, shape_k, shape_n)
  K-major  → inner=K,  outer=N
  MN-major → inner=N,  outer=K
        │
        ▼
JIT kernel instantiation
  template <Major kMajorB>
  sm100_bf16_asym_gemm<..., kMajorB>
        │
        ├─ tma_copy<BLOCK_K, LOAD_BLOCK_N>  (K-major)
        │  tma_copy<LOAD_BLOCK_N, BLOCK_K>  (MN-major)
        │
        └─ get_umma_desc_stride_k()          sm100_utils.cuh:83
             K-major  → stride_k = 1
             MN-major → stride_k = BLOCK_MN_ATOM
```

All downstream differences (swizzle bytes, UMMA SBO/LBO, descriptor advance step) are fully baked into the two JIT-compiled kernel variants. **Both variants exist and are cached independently.**

---

## Memory Layout Analysis: Is Non-Contiguous Access a Performance Problem?

### Forward: `gate_up_cpu [E, 2I, H]` K-major

Physical layout (per expert):

```
row 0 → [w_00, w_01, …, w_0,H-1]   ← H contiguous BF16 elements
row 1 → [w_10, w_11, …, w_1,H-1]
…
row 2I-1 → [w_{2I-1,0}, …, w_{2I-1,H-1}]
```

TMA tile `(BLOCK_N=128, BLOCK_K=64)` fetches 128 K-contiguous elements per row — high cache-line utilisation.

```
TMA inner dim = K = H     (stride-1, contiguous) ✓
TMA outer dim = N = 2I    (stride H)
outer_stride  = H         (row width of physical matrix)
```

### Backward: same physical buffer, transposed logical view

We want to compute `dX = d_gate_up [T, 2I] @ W_gate_up [2I, H]`.  
AsymGEMM is an `NT` kernel (computes `A @ B.T`), so we need `B = W_gate_up.T [H, 2I]`.

Physical memory is unchanged. We just change what the TMA descriptor says is "inner" and "outer":

```
"Column" j of the transposed view (all H elements for fixed 2I index j):
   physical offsets: j·H,  j·H+1,  j·H+2,  …,  j·H+(H-1)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                     H consecutive elements — contiguous ✓
```

TMA tile `(BLOCK_N'=H_block, BLOCK_K'=2I_block)` fetches H-contiguous elements per column:

```
TMA inner dim = N' = H    (stride-1, contiguous) ✓   ← SAME contiguity guarantee
TMA outer dim = K' = 2I   (stride H)
outer_stride  = H         (identical to forward: physical row width)
```

**Both forward and backward have a stride-1 inner dimension.** The TMA engine reads full cache lines in both cases. The only structural difference is which axis is "inner" (K vs H).

### Where the Two Descriptors Differ

| Parameter                   | Forward (K-major)             | Backward (MN-major)              |
|-----------------------------|-------------------------------|----------------------------------|
| `gmem_inner_dim`            | `K = H`                       | `N = H` (same value, axis role swaps) |
| `gmem_outer_dim`            | `N = 2I`                      | `K = 2I` (same value, axis role swaps) |
| `outer_stride`              | `H`                           | `H` (identical physical stride)  |
| TMA copy template           | `<BLOCK_K, LOAD_BLOCK_N>`     | `<LOAD_BLOCK_N, BLOCK_K>`        |
| TMA copy indices            | `(k_idx, n_idx)`              | `(n_idx, k_idx)`                 |
| UMMA `stride_k`             | `1`                           | `BLOCK_MN_ATOM`                  |
| UMMA SBO                    | `8 × BLOCK_K × 2`             | `8 × BLOCK_MN_ATOM × 2`         |
| UMMA LBO                    | `128B` or `0`                 | `BLOCK_K × BLOCK_MN_ATOM × 2`   |

**The outer_stride value is the same for both.** The only real difference is the role assignment (which axis is inner vs outer) and the downstream SMEM layout for UMMA.

### Bandwidth Impact on NVLink-C2C

NVLink-C2C characteristics relevant here:
- Bandwidth: ~900 GB/s (host→device)  
- Granularity: 64-byte cache lines from CPU DRAM
- Latency: ~1 µs round-trip

Both layouts produce the **same byte count transfer** from CPU DRAM. The inner dimension is contiguous in both cases (stride-1 guarantees full cache-line packing). NVLink-C2C bandwidth is not sensitive to the axis labelling — only to contiguity of the inner dimension.

The only potential difference is in **SMEM bank conflict behaviour** (UMMA SBO/LBO differ), but both values are pre-tuned in the JIT-compiled kernel variants. No manual tuning is required.

**Expected performance**: backward ≈ forward in bandwidth, because the physical access pattern (contiguous inner tiles, strided outer) is structurally identical.

---

## Proposed API

### Current (Phase 4)

```python
# weight_manager.py
gate_up_T = gate_up_cpu.permute(0, 2, 1)   # non-contiguous view → MN-major auto-detected

# autograd_fn.py
asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(
    a, gate_up_T, out, offsets, experts, list_size, "mnk")   # MN-major from strides
```

### Proposed (explicit `transpose_b` flag)

```python
# weight_manager.py
# No gate_up_T needed at all — removed from PinnedExpertWeights

# moe_ops.py (backward)
asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(
    a, gate_up_cpu, out, offsets, experts, list_size, "mnk",
    transpose_b=True)    # kernel builds transposed TMA desc internally
```

`gate_up_cpu [E, 2I, H]` is now passed for both forward (`transpose_b=False`, default) and backward (`transpose_b=True`). No permuted view, no stride-based dispatch.

---

## Implementation Plan

### Step 1 — C++ wrapper: add `transpose_b` flag

**File**: `csrc/apis/gemm.hpp`

Current dispatch (line ~121):
```cpp
const auto major_b = get_major_type_ab(b);
```

Add an override path:
```cpp
// New optional argument to the top-level dispatch function:
// bool transpose_b = false

const auto major_b = transpose_b
    ? cute::UMMA::Major::MN   // build MN-major descriptor regardless of strides
    : get_major_type_ab(b);   // existing auto-detect path (unchanged)
```

`get_major_type_ab()` is only called in the `!transpose_b` branch, so existing code paths are completely unaffected.

### Step 2 — TMA descriptor: derive dimensions from physical layout

When `transpose_b=True`, the tensor is physically `[N_phys, K_phys]` K-major, but we want to build an MN-major descriptor for `[K_phys, N_phys]`.

**File**: `csrc/jit_kernels/impls/sm100_bf16_asym_gemm.hpp` (around line 160)

Current:
```cpp
const auto& tensor_map_b = make_tma_b_desc(
    major_b, b, n, k,
    block_n, block_k,
    static_cast<int>(b.stride(get_non_contiguous_dim(major_b))),
    num_groups, swizzle_b_mode);
```

With `transpose_b`:
```cpp
// When transpose_b, shape_n/shape_k are swapped relative to the physical tensor:
//   physical: b has shape [n_phys=n, k_phys=k], K-major, outer_stride = k
//   logical:  treat as  [n_logical=k, k_logical=n], MN-major, outer_stride = k (same physical row width)
const int tma_n          = transpose_b ? k : n;          // logical N for this GEMM
const int tma_k          = transpose_b ? n : k;          // logical K for this GEMM
const int tma_outer_stride = transpose_b
    ? static_cast<int>(b.stride(0) / b.size(1))          // = k_phys (physical row width)
    : static_cast<int>(b.stride(get_non_contiguous_dim(major_b)));

const auto& tensor_map_b = make_tma_b_desc(
    major_b, b, tma_n, tma_k,
    block_n, block_k,
    tma_outer_stride,
    num_groups, swizzle_b_mode);
```

Because `outer_stride` = physical row width = `K_phys = H` in both directions, the `transpose_b` branch computes the same numerical value as the existing auto-detect path. The change is that we bypass stride-based detection and make the intent explicit.

### Step 3 — Python bindings: expose `transpose_b`

**File**: `csrc/bindings.cpp` (PyBind11 binding for `m_grouped_bf16_asym_gemm_nt_contiguous`)

```cpp
m.def("m_grouped_bf16_asym_gemm_nt_contiguous",
      &m_grouped_bf16_asym_gemm_nt_contiguous,
      py::arg("a"), py::arg("b"), py::arg("out"),
      py::arg("offsets"), py::arg("expert_ids"), py::arg("list_size"),
      py::arg("layout"),
      py::arg("transpose_b") = false);      // ← new keyword argument, default false
```

All existing callers (with 7 positional args) are unaffected.

### Step 4 — `moe_ops.py`: route forward vs backward

```python
def _call_asym_gemm_nt(a, b_cpu, offsets, experts, list_size, *, transpose_b=False):
    M, K = a.shape
    N = b_cpu.shape[2] if transpose_b else b_cpu.shape[1]   # [E, K_phys, N_phys] vs [E, N, K]
    out = torch.empty(M, N, dtype=torch.bfloat16, device=a.device)
    asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(
        a, b_cpu, out, offsets, experts, list_size, "mnk",
        transpose_b=transpose_b)
    return out


def moe_forward_inner(permuted_hidden, gate_up_cpu, down_cpu, offsets, experts, list_size):
    gate_up_out = _call_asym_gemm_nt(permuted_hidden, gate_up_cpu, offsets, experts, list_size)
    gate_act, up_act = gate_up_out.chunk(2, dim=-1)
    intermediate = F.silu(gate_act) * up_act
    expert_out = _call_asym_gemm_nt(intermediate, down_cpu, offsets, experts, list_size)
    return expert_out, gate_act, up_act, intermediate


def moe_backward_dx(grad_out, gate_act, up_act,
                    gate_up_cpu, down_cpu,          # ← same tensors as forward
                    offsets, experts, list_size):
    # transpose_b=True: kernel builds MN-major TMA desc from K-major storage
    d_intermediate = _call_asym_gemm_nt(grad_out, down_cpu,    offsets, experts, list_size,
                                        transpose_b=True)
    d_gate_up = _dsilu_gate_mul_up(d_intermediate, gate_act, up_act)
    d_hidden   = _call_asym_gemm_nt(d_gate_up, gate_up_cpu, offsets, experts, list_size,
                                    transpose_b=True)
    return d_hidden
```

### Step 5 — `weight_manager.py`: remove transposed fields

```python
@dataclass
class PinnedExpertWeights:
    gate_up: torch.Tensor   # [E, 2I, H]  CPU pinned BF16  (used for both forward and backward)
    down:    torch.Tensor   # [E, H, I]   CPU pinned BF16  (used for both forward and backward)
    num_experts:  int
    hidden:       int
    intermediate: int
    # gate_up_T and down_T are gone — no second copy, no permuted view
```

`extract_and_pin_experts()` drops the two `.permute()` lines entirely.

### Step 6 — `autograd_fn.py`: remove T tensors from ctx

```python
# forward: no gate_up_T_cpu / down_T_cpu stored in ctx
ctx.gate_up_cpu = gate_up_cpu   # same tensor used for backward
ctx.down_cpu    = down_cpu

# backward: pass originals with transpose_b handled inside moe_ops
d_perm_hidden = moe_backward_dx(
    grad_expert, gate_act, up_act,
    ctx.gate_up_cpu, ctx.down_cpu,
    offsets, experts_t, list_size,
)
```

The `asym_gemm_moe` public signature drops `gate_up_T_cpu` and `down_T_cpu`:

```python
def asym_gemm_moe(hidden_states, gate_up_cpu, down_cpu,
                  routing_weights, selected_experts, num_experts, top_k):
    return AsymGEMMMoEFunction.apply(
        hidden_states, gate_up_cpu, down_cpu,
        routing_weights, selected_experts, num_experts, top_k)
```

`moe_wrapper.py` and tests updated accordingly (remove `gate_up_T` / `down_T` arguments).

---

## Comparison: Phase 4 View vs Proposed Flag

| Concern                          | Phase 4 (permuted view)                       | Proposed (`transpose_b` flag)                     |
|----------------------------------|-----------------------------------------------|---------------------------------------------------|
| CPU DRAM usage                   | 1× (same storage)                             | 1× (same storage)                                 |
| Python-side objects              | 2 tensor handles per weight (original + view) | 1 tensor handle per weight                        |
| Kernel code path                 | MN-major (auto-detected from strides)         | MN-major (explicit flag)                          |
| TMA descriptor built             | Identical                                     | Identical                                         |
| Physical memory access pattern   | Identical                                     | Identical                                         |
| Expected bandwidth               | Same                                          | Same                                              |
| Requires kernel modification     | No                                            | Yes (~20 lines in gemm.hpp + bindings)            |
| API clarity                      | Implicit (strides encode intent)              | Explicit (`transpose_b=True` documents intent)    |
| Risk of misuse                   | Low (auto-detect is robust)                   | Low (opt-in flag with safe default)               |
| JIT cache hit rate               | Same (MN-major variant already compiled)      | Same (MN-major variant reused)                    |

---

## Performance Expectation

The TMA descriptor difference reduces to:

- **Forward (K-major)**: `tma_copy<BLOCK_K=64, LOAD_BLOCK_N=128>` with `stride_k=1`  
- **Backward (MN-major)**: `tma_copy<LOAD_BLOCK_N=128, BLOCK_K=64>` with `stride_k=BLOCK_MN_ATOM`

The total bytes read from CPU DRAM per GEMM are identical. The inner TMA dimension is stride-1 in both cases, so NVLink-C2C cache-line utilisation is the same.

The UMMA stride difference (`1` vs `BLOCK_MN_ATOM`) affects how the tensor-core descriptor advances through shared memory — this is handled by the pre-compiled MN-major kernel variant and is already tuned. No performance cliff is expected.

**Any latency difference** will come from the UMMA SBO/LBO mismatch vs the optimal shared-memory layout for a given `BLOCK_N` and swizzle mode. The JIT compiler selects the swizzle automatically from the tile configuration, so both variants should be near-optimal.

We recommend benchmarking with the SM100 hardware profiler (`ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld`) to confirm NVLink-C2C utilisation is the same for both directions.

---

## File Change Summary

| File                                                         | Change                                          |
|--------------------------------------------------------------|-------------------------------------------------|
| `AsymGEMM_main/csrc/apis/gemm.hpp`                          | Add `bool transpose_b = false` → override `major_b` |
| `AsymGEMM_main/csrc/jit_kernels/impls/sm100_bf16_asym_gemm.hpp` | Derive `tma_n / tma_k / tma_outer_stride` from flag |
| `AsymGEMM_main/csrc/bindings.cpp`                           | Expose `transpose_b` keyword arg (default false) |
| `asym_sft/moe_ops.py`                                       | `_call_asym_gemm_nt(..., transpose_b=)`, update `moe_backward_dx` signature |
| `asym_sft/weight_manager.py`                                | Remove `gate_up_T`, `down_T` from `PinnedExpertWeights` and from `extract_and_pin_experts` |
| `asym_sft/autograd_fn.py`                                   | Remove T-tensor args from ctx, update `asym_gemm_moe` signature |
| `asym_sft/moe_wrapper.py`                                   | Remove `weights.gate_up_T / weights.down_T` from `forward()` call |
| `tests/test_asym_moe.py`                                    | Update `make_expert_weights`, `asym_gemm_moe` call sites |
| `benchmark_kernels.py`                                      | Update backward benchmark section               |

---

## Recommendation

**Implement the `transpose_b` flag** if you want the cleanest long-term API (one weight tensor, one arg). The kernel change is small (~20 lines across two C++ files + bindings), the TMA descriptor produced is provably identical to Phase 4, and the performance impact is negligible.

If kernel modification is out of scope, **Phase 4 (permuted view) is already correct** — the auto-detected MN-major path exercises the same JIT-compiled kernel variant. The only thing the flag adds is explicitness.
