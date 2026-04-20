# CUDA Graph Compatibility Plan for `offsets`/`experts` Construction

## 1. Problem Statement

`build_offsets_experts_from_masked_m` and `build_offsets_experts_from_m_indices_pairs`
(defined in `tests/test_fp8.py`) perform host-side work that prevents CUDA graph capture.
In an end-to-end system such as sglang the same logic runs every forward step, so its
CPU overhead and D2H synchronisation become a hard blocker for graph-based inference.

---

## 2. Root Cause Analysis

### 2.1 `build_offsets_experts_from_masked_m`

```python
for g in range(num_groups):
    v = masked_m[g].item()          # ❶ D2H sync + CPU–GPU barrier
    if v > 0:
        start = g * max_m
        end   = start + ceil(v, block_m) * block_m
        offsets.append(start); offsets.append(end)
        experts.append(g)
experts.append(-1)
return (torch.tensor(offsets, ...), torch.tensor(experts, ...), len(experts))
```

| # | CPU-blocking operation | Why it breaks CUDA graphs |
|---|------------------------|--------------------------|
| ❶ | `masked_m[g].item()` inside a Python `for` loop | Forces a device→host copy + CUDA stream sync on every iteration; graph capture is impossible while a sync is in flight |
| ❷ | Conditional `if v > 0` on the CPU-side value | Produces a variable-length `offsets`/`experts` list whose length is unknown until the values are read |
| ❸ | `torch.tensor(offsets, ...)` at the end | Allocates a new tensor with a shape determined at runtime → graph capture records a fixed-shape allocation that is wrong for different sparsity patterns |
| ❹ | `list_size = len(experts)` (Python int) | Feeds directly into the C++ launch as `grid_y = list_size - 1`; a graph records a fixed grid dimension that no longer matches when the active-expert count changes |

### 2.2 `build_offsets_experts_from_m_indices_pairs`

```python
starts  = nonzero(change) + 1           # GPU: OK
...
for start, end, eid in zip(
        segment_starts.tolist(),        # ❺ D2H copy of entire segment arrays
        segment_ends.tolist(),
        segment_experts.tolist()):
    if eid == -1: continue              # ❻ CPU conditional on GPU-derived value
    start_padded = (start // block_m) * block_m
    end_padded   = ((end + block_m - 1) // block_m) * block_m
    offsets.append(start_padded); offsets.append(end_padded)
    experts.append(eid)
experts.append(-1)
return (torch.tensor(offsets, ...), torch.tensor(experts, ...), len(experts))
```

Same failure classes as above; `.tolist()` pulls the full segment arrays to the CPU and
`list_size` is again a runtime-variable Python integer.

### 2.3 `list_size` as a Grid Dimension

The C++ dispatch in `sm100_m_grouped_fp8_asym_gemm_masked_1d1d` / `_contiguous_1d1d` uses:

```cpp
.launch_args = LaunchArgs(
    {ceil_div(n, config.block_n), list_size - 1},   // grid Y = active experts
    ...)
```

CUDA graphs record the exact grid dimensions at capture time.  If `list_size` changes
between steps (because a different set of experts is active), every captured kernel node
has the wrong `gridDim.y` → correctness failure or hang.

---

## 2.4 Comparison: How Reference Implementations Handle This

### DeepGEMM (GPU-resident scheduler, no offset builder)

DeepGEMM solves an equivalent problem by **eliminating the offset/expert builder entirely**.
Its kernel scheduler (`deep_gemm/include/deep_gemm/scheduler/gemm.cuh`) receives `masked_m`
directly as a GPU pointer (`grouped_layout`) and iterates groups on-device:

```cpp
// Scheduler::get_next_block (runs on GPU, reads grouped_layout in-place)
num_m_blocks = math::ceil_div(static_cast<uint32_t>(grouped_layout[current_group_idx]), BLOCK_M);
```

The Python call site passes `masked_m` as-is — no `.item()`, no `torch.tensor(...)`:

```python
# DeepGEMM C++ API (csrc/apis/gemm.hpp)
.grouped_layout = masked_m.data_ptr()   // GPU pointer → GPU kernel reads directly
```

The grid is a **persistent 1-D kernel** launched with `{num_sms, 1}` blocks.  `num_sms` is a
hardware constant, so the grid dimension is the same on every replay.  All group-iteration
state (`current_group_idx`, `current_m_cumsum`, …) is held in registers — nothing comes back
to the host.

**Key lesson**: if the kernel scheduler itself can handle the sparse routing, the Python-level
offset/expert builder is completely unnecessary.  AsymGEMM currently uses a 2-D persistent
launch (`{ceil_div(N, block_n), list_size-1}`) that exposes `list_size` to the host; the GPU
builder approach in §4 is the minimum change required to fix that.

### Flash-Attention (capture/replay pattern)

Flash-attention's `capture_graph` (in `flash_attn/utils/generation.py`) demonstrates the
canonical capture/replay idiom:

```python
# Step 1 — Warmup on a side stream (avoids polluting the main stream's state)
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())      # side stream waits for any in-flight work
with torch.cuda.stream(s):
    for _ in range(n_warmups):
        logits = model(input_ids, ...)          # runs outside capture, warms caches/allocators
    s.synchronize()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()             # NCCL graph-mixing compatibility (important for MoE)
torch.cuda.current_stream().wait_stream(s)      # main stream waits for warmup to finish

# Step 2 — Capture
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, pool=mempool):     # pool: shared handle for memory reuse
    logits = model(input_ids, ...)

# Step 3 — Replay closure: copy fresh data into static tensors, then replay
def run(new_input_ids, new_position_ids, seqlen):
    inference_params.lengths_per_sample[:] = seqlen
    input_ids.copy_(new_input_ids)              # in-place copy into static buffer
    position_ids.copy_(new_position_ids)
    graph.replay()
    return logits.clone()
```

The `mempool` is created once with `torch.cuda.graphs.graph_pool_handle()` and shared across
all captures for the same model.  Sharing a pool allows CUDA to reuse the same physical memory
blocks across multiple captured graphs, avoiding redundant allocations and fragmentation.

**Key lessons for AsymGEMM**:
- Always warm up on a **side stream** before capture so temporary allocations do not appear
  inside the captured region.
- Wait for the warmup stream to finish with `current_stream().wait_stream(s)` before entering
  `torch.cuda.graph(...)`.
- Add `torch.distributed.barrier()` between warmup and capture when running under a NCCL
  process group (sglang MoE uses tensor/expert parallelism).
- Use `graph_pool_handle()` for all captures that belong to the same model instance so that
  their GPU memory footprints do not stack.
- The replay closure must only use **in-place** writes (`copy_`, `fill_`, slice assignment)
  into the pre-allocated static tensors that were live at capture time.

---

## 3. CUDA Graph Requirements

For a computation to be re-playable inside a CUDA graph the following must hold:

1. **No host↔device synchronisation** during the replayed region.
2. **Fixed grid dimensions** — `gridDim` must be the same on every replay.
3. **Fixed buffer shapes** — tensor allocations are recorded once; the same pointers and
   sizes are reused on replay.
4. **No Python or CPU arithmetic** that touches GPU-resident values at runtime.

All four requirements are violated by the current helpers.

---

## 4. Proposed Solution

The core idea is a **fixed-size dense layout**: always allocate buffers for *all* `num_groups`
expert slots, populate them entirely on the GPU, and let the GEMM kernel skip slots whose
M range is empty (`m_start == m_end`).  This makes both the buffer shape and the grid
dimensions constants that depend only on the model configuration, not on the routing outcome.

### 4.1 New Buffer Layout

| Buffer | Old shape (compact) | New shape (dense, fixed) |
|--------|--------------------|-----------------------------|
| `offsets` | `2 × num_active` (variable) | `2 × num_groups` (constant) |
| `experts` | `num_active + 1` (variable) | `num_groups + 1` (constant) |
| `list_size` | `num_active + 1` (variable Python int) | `num_groups + 1` (constant Python int) |

Active groups carry their real `[start, end)` pair.  Inactive groups carry an **empty
range**: `offsets[2g] = offsets[2g+1] = g * max_m` (for masked) or `offsets[2g] =
offsets[2g+1] = 0` (for contiguous).  `experts[g] = g` for all `g`; `experts[num_groups]
= -1` stays as the terminator at a fixed position.

`list_size` becomes `num_groups + 1` unconditionally — a compile-time constant for a
given model — so the grid dimension `grid_y = num_groups` is fixed.

### 4.2 Part A — New CUDA Kernel: `build_offsets_experts_masked`

**Location**: new file `csrc/indexing/build_offsets.cuh` + exposed via `csrc/indexing/main.cu`

```cuda
// One thread block, one thread per group (num_groups ≤ 1024 for typical MoE).
__global__ void build_offsets_experts_masked_kernel(
        const int*    masked_m,     // [num_groups]  actual token counts per group
        uint32_t*     offsets,      // [2*num_groups] output offset pairs
        int*          experts,      // [num_groups+1] output expert IDs + terminator
        const int     num_groups,
        const int     max_m,
        const int     block_m)
{
    int g = threadIdx.x + blockIdx.x * blockDim.x;
    if (g >= num_groups) return;

    int v     = masked_m[g];
    int start = g * max_m;
    int end   = (v > 0) ? start + ((v + block_m - 1) / block_m) * block_m
                        : start;          // empty range: end == start

    offsets[2*g]     = static_cast<uint32_t>(start);
    offsets[2*g + 1] = static_cast<uint32_t>(end);
    experts[g]       = g;                 // identity; kernel uses expert_id for B/D indexing

    // Write terminator at the fixed last position (one extra thread or the last thread)
    if (g == num_groups - 1)
        experts[num_groups] = -1;
}
```

**Python binding** (added to `csrc/python_api.cpp` via `csrc/indexing/main.cu`):

```python
asym_gemm.build_offsets_experts_masked(
    masked_m,          # (G,) int32 CUDA tensor
    offsets_buf,       # (2*G,) int32 CUDA tensor — pre-allocated, persistent
    experts_buf,       # (G+1,) int32 CUDA tensor — pre-allocated, persistent
    num_groups,        # Python int (model constant)
    max_m,             # Python int (model constant)
    block_m=128,       # Python int (model constant)
)
```

Because `offsets_buf` and `experts_buf` are **pre-allocated once** (during model
initialisation) and reused on every step, no new allocation happens inside the graph.

### 4.3 Part B — New CUDA Kernel: `build_offsets_experts_contiguous`

For the contiguous case `m_indices` is a sorted, 1-D token→expert assignment array.
Each unique contiguous run of expert ID `g` maps to one entry in the output.  With
sorted routing (standard MoE), every expert appears at most once.

**Algorithm** (runs entirely on GPU):

1. **Per-expert binary search** — for each `g ∈ [0, num_groups)` use two
   `torch.searchsorted` calls (or a custom kernel) to find `[first_token, last_token)`.
2. **Pad to `block_m`** — apply floor/ceil alignment to `start`/`end`.
3. **Write to fixed-size buffers** — empty experts get `start == end`.

```cuda
__global__ void build_offsets_experts_contiguous_kernel(
        const int* __restrict__ m_indices,   // [M] sorted expert IDs (−1 = invalid)
        uint32_t*               offsets,     // [2*num_groups]
        int*                    experts,     // [num_groups+1]
        const int               num_groups,
        const int               M,
        const int               block_m)
{
    int g = threadIdx.x + blockIdx.x * blockDim.x;
    if (g >= num_groups) return;

    // Binary search for first and last occurrence of expert g.
    // (Replace with device-side lower/upper_bound; omitted for brevity.)
    int lo = lower_bound(m_indices, M, g);
    int hi = upper_bound(m_indices, M, g);

    uint32_t start = (lo == hi) ? 0u
                                : static_cast<uint32_t>((lo / block_m) * block_m);
    uint32_t end   = (lo == hi) ? 0u
                                : static_cast<uint32_t>(((hi + block_m - 1) / block_m) * block_m);

    offsets[2*g]     = start;
    offsets[2*g + 1] = end;
    experts[g]       = g;
    if (g == num_groups - 1)
        experts[num_groups] = -1;
}
```

For unsorted or non-unique routing (one expert can appear multiple times), a parallel
histogram + prefix-sum pass is needed first to convert `m_indices` into a per-group count
array before applying the same pattern above.

### 4.4 Part C — GEMM Kernel Early-Exit Guard

With the dense layout, `grid_y = num_groups` always includes slots for inactive experts.
Those blocks must exit quickly without touching TMA barriers that other blocks depend on.

**Change in `sm100_fp8_asym_gemm_1d1d_impl`** (and the BF16 / FP4 counterparts):

```cuda
// After cluster_sync (line ~212) and scheduler construction (line ~215):
auto scheduler = asymScheduler<...>(shape_m, shape_n, experts, offsets);

// ── NEW ──────────────────────────────────────────────────────────────────────
// Empty-range guard: no tiles to process for this block → return immediately.
// Both CTAs in a 2-CTA cluster read the same blockIdx.y, so they compute the
// same m_start/m_end and will both hit this branch consistently.
if (scheduler.m_start >= scheduler.m_end)
    return;
// ─────────────────────────────────────────────────────────────────────────────
```

This guard is safe because:
- The preceding `cluster_sync()` has already resolved all cluster barriers.
- `m_start` and `m_end` are computed from the same `offsets[blockIdx.y*2]` by all CTAs
  that share the same `blockIdx.y`; they will unanimously agree on the early exit.
- No subsequent TMA or TMEM barrier has been initialised yet at this point, so no barrier
  can be left in an un-arrived state.

### 4.5 Part D — C++ API Update

The launch grid in `sm100_m_grouped_fp8_asym_gemm_masked_1d1d` and
`sm100_m_grouped_fp8_asym_gemm_contiguous_1d1d` currently uses `list_size - 1` for
`grid_y`.  After this change `list_size` is always `num_groups + 1`, so the expression
simplifies to the model-constant `num_groups`:

```cpp
// Before
.launch_args = LaunchArgs({ceil_div(n, config.block_n), list_size - 1}, ...)

// After (list_size is now always num_groups + 1)
.launch_args = LaunchArgs({ceil_div(n, config.block_n), num_groups}, ...)
```

No change is needed to the `offsets` / `experts` pointer passing — the kernel already
reads them from GPU memory.

The Python-facing `list_size` parameter can be deprecated or, to preserve backward
compatibility, validated against `num_groups + 1` with an assertion.

### 4.6 Part E — Updated Python Helpers in Tests

Replace the two CPU-side builders with thin wrappers that call the GPU kernels and
manage the persistent pre-allocated buffers:

```python
class OffsetExpertBuffer:
    """Pre-allocated, persistent GPU buffers for CUDA-graph-friendly scheduling."""

    def __init__(self, num_groups: int, device='cuda'):
        self.num_groups = num_groups
        # Fixed-shape, allocated once at model init time.
        self.offsets = torch.empty(2 * num_groups,   dtype=torch.int32, device=device)
        self.experts = torch.empty(num_groups + 1,   dtype=torch.int32, device=device)
        self.list_size = num_groups + 1              # constant Python int

    def build_masked(self, masked_m: torch.Tensor, max_m: int, block_m: int = 128):
        """Pure GPU build — safe inside CUDA graph capture."""
        asym_gemm.build_offsets_experts_masked(
            masked_m, self.offsets, self.experts,
            self.num_groups, max_m, block_m)
        return self.offsets, self.experts, self.list_size

    def build_contiguous(self, m_indices: torch.Tensor, block_m: int = 128):
        """Pure GPU build — safe inside CUDA graph capture."""
        asym_gemm.build_offsets_experts_contiguous(
            m_indices, self.offsets, self.experts,
            self.num_groups, m_indices.numel(), block_m)
        return self.offsets, self.experts, self.list_size
```

Usage at the call site becomes:

```python
# One-time init (outside graph region)
buf = OffsetExpertBuffer(num_groups, device='cuda')

# Inside graph capture / every step
offsets, experts, list_size = buf.build_masked(masked_m, max_m)
asym_gemm.m_grouped_fp8_asym_gemm_nt_masked(
    a, b, d, offsets, experts, list_size, expected_m)
```

---

## 5. Implementation Steps

| Step | File(s) to change | Description |
|------|--------------------|-------------|
| 1 | `csrc/indexing/main.cu` (new kernel bodies) | Implement `build_offsets_experts_masked_kernel` and `build_offsets_experts_contiguous_kernel` as `__global__` functions |
| 2 | `csrc/indexing/main.cu` | Add PyTorch C++ wrapper functions that validate tensor shapes/dtypes and launch the kernels |
| 3 | `csrc/python_api.cpp` | Register the two new Python bindings via `pybind11` |
| 4 | `asym_gemm/include/asym_gemm/impls/sm100_fp8_asym_gemm_1d1d.cuh` | Add the empty-range early-exit guard after scheduler construction |
| 5 | Same for `sm100_bf16_asym_gemm.cuh` and `sm100_fp4_asym_gemm_1d1d.cuh` | Identical early-exit guard in the BF16 and FP4 GEMM kernels |
| 6 | `csrc/jit_kernels/impls/sm100_fp8_asym_gemm_1d1d.hpp` | Change `list_size - 1` → `num_groups` in the `LaunchArgs` constructor; add assertion `list_size == num_groups + 1` |
| 7 | Same for `sm100_bf16_asym_gemm.hpp` and `sm100_fp4_asym_gemm_1d1d.hpp` | Same grid-dim change |
| 8 | `tests/test_fp8.py`, `tests/test_bf16.py`, `tests/test_nvfp4.py` | Replace `build_offsets_experts_from_masked_m` / `build_offsets_experts_from_m_indices_pairs` calls with `OffsetExpertBuffer` wrappers |
| 9 | New file `tests/test_cuda_graph.py` | End-to-end CUDA graph capture and replay test (see §6) |

---

## 6. Testing Strategy

### 6.1 Correctness Test (new)

`tests/test_cuda_graph.py` — verify that the GPU builder produces the same
`offsets`/`experts` as the old CPU builders for a range of `masked_m` patterns:

```python
def test_build_offsets_masked_equivalence():
    for num_groups, max_m in [(6, 4096), (32, 4096)]:
        masked_m = torch.randint(0, max_m // 2 + 1, (num_groups,), device='cuda', dtype=torch.int32)
        # Reference (old CPU path)
        ref_offsets, ref_experts, ref_ls = build_offsets_experts_from_masked_m(masked_m, num_groups, max_m)

        # New GPU path
        buf = OffsetExpertBuffer(num_groups)
        gpu_offsets, gpu_experts, gpu_ls = buf.build_masked(masked_m, max_m)

        # Dense buffers include inactive slots; compare only active entries
        # (filter by non-empty ranges in gpu_offsets)
        assert gpu_ls == num_groups + 1
        for g in range(num_groups):
            if masked_m[g] > 0:
                assert gpu_offsets[2*g] == ref_offsets[...],   "start mismatch"
                assert gpu_offsets[2*g+1] == ref_offsets[...], "end mismatch"
```

### 6.2 CUDA Graph Capture Test

The capture pattern below mirrors the flash-attention `capture_graph` idiom (side-stream
warmup → optional distributed barrier → capture → replay closure):

```python
def test_cuda_graph_masked_gemm():
    num_groups, max_m, n, k = 32, 4096, 4096, 7168
    # ... generate a (FP8), b (FP8), masked_m (int32 on CUDA) ...

    buf = OffsetExpertBuffer(num_groups)
    d   = torch.empty((num_groups, max_m, n), dtype=torch.bfloat16, device='cuda')

    # ── Warmup on a side stream (flash-attention pattern) ────────────────────
    # Running warmup outside the main stream prevents temporary allocations
    # (e.g. inside JIT compilation or cuBLAS handle initialisation) from being
    # recorded into the captured graph.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            offsets, experts, ls = buf.build_masked(masked_m, max_m)
            asym_gemm.m_grouped_fp8_asym_gemm_nt_masked(a, b, d, offsets, experts, ls, max_m)
        s.synchronize()
        # Required when running under NCCL (e.g. tensor-parallel sglang):
        # NCCL_GRAPH_MIXING_SUPPORT=0 requires that graph-captured and
        # non-captured NCCL operations do not overlap.
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    torch.cuda.current_stream().wait_stream(s)

    # ── Capture ───────────────────────────────────────────────────────────────
    # Use a shared pool (graph_pool_handle) so that all captures for this model
    # instance reuse the same physical GPU memory rather than stacking.
    mempool = torch.cuda.graphs.graph_pool_handle()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=mempool):
        offsets, experts, ls = buf.build_masked(masked_m, max_m)
        asym_gemm.m_grouped_fp8_asym_gemm_nt_masked(a, b, d, offsets, experts, ls, max_m)

    # ── Replay closure ────────────────────────────────────────────────────────
    # New routing data is written in-place into the persistent masked_m buffer;
    # the output d is also a persistent buffer — clone it if the caller needs
    # a stable copy across replays.
    def run(new_masked_m: torch.Tensor) -> torch.Tensor:
        masked_m.copy_(new_masked_m)   # in-place update of the static buffer
        g.replay()
        return d.clone()

    # Verify: replay with a different routing and compare to eager execution
    new_masked_m = torch.randint(0, max_m // 2 + 1, (num_groups,), device='cuda', dtype=torch.int32)
    graph_out = run(new_masked_m)

    # Eager reference (using the same pre-allocated buffers, no graph)
    ref_offsets, ref_experts, ref_ls = buf.build_masked(new_masked_m, max_m)
    asym_gemm.m_grouped_fp8_asym_gemm_nt_masked(a, b, d, ref_offsets, ref_experts, ref_ls, max_m)
    assert torch.allclose(graph_out, d, atol=1e-2), "Graph replay output differs from eager"
```

**Why side-stream warmup matters**: on first call, DeepGEMM's JIT compiler (and PyTorch's
caching allocator) may perform one-time work (kernel compilation, cuBLAS handle init) that
allocates temporary host/device memory.  These allocations must not be visible inside
`torch.cuda.graph(...)`, otherwise CUDA will record them as fixed-address allocations that
are wrong on the next replay.  Running the warmup iterations outside the capture region
prevents this.

**`graph_pool_handle()` and memory sharing**: when a model layer is captured as multiple
graphs (e.g. one per `num_tokens` bucket in decode), passing the same `pool` to every
`torch.cuda.graph(g, pool=mempool)` call lets CUDA assign overlapping physical memory to
graphs that never execute concurrently, reducing peak GPU memory by up to the number of
captured graphs.

### 6.3 Performance Regression Test

Run the existing `bench_kineto` benchmarks from `test_fp8.py` before and after; confirm
that latency for active-expert configurations is not degraded, and measure the overhead
introduced by empty-slot blocks at varying sparsity levels (0 %, 25 %, 50 % inactive
experts).

---

## 7. Backward Compatibility

- The existing `build_offsets_experts_from_masked_m` and
  `build_offsets_experts_from_m_indices_pairs` Python functions can be kept **unchanged**
  for non-graph execution paths and for unit tests that exercise the CPU path.
- The C++ GEMM API signature is **unchanged** (`offsets`, `experts`, `list_size` remain
  the same three parameters).
- The only observable difference for callers is that `list_size` is now always
  `num_groups + 1` (previously it could be smaller).  An assertion is added to catch
  any caller still passing a compact `list_size`.
- The early-exit guard in the GEMM kernel is a pure addition; it does not affect the
  output for non-empty blocks.

---

## 8. Alternative: Eliminate the Offset/Expert Layer Entirely (DeepGEMM Approach)

DeepGEMM avoids the offset/expert indirection layer altogether.  Its SM100 masked-GEMM
kernel (`sm100_m_grouped_fp8_fp4_gemm_masked_1d1d`) launches a **1-D persistent grid**
with `{num_sms, 1}` blocks and passes `masked_m.data_ptr()` directly as the
`grouped_layout` pointer.  The device-side `Scheduler` struct reads token counts from GPU
memory and walks group boundaries entirely in registers:

```cpp
// deep_gemm/include/deep_gemm/scheduler/gemm.cuh
num_m_blocks = math::ceil_div(static_cast<uint32_t>(grouped_layout[current_group_idx]), BLOCK_M);
```

Because the grid is `{num_sms, 1}` (a hardware constant) and `grouped_layout` is a fixed
GPU pointer, this launch is trivially CUDA-graph-safe with no additional infrastructure.

This design is more invasive for AsymGEMM because our current launch uses a 2-D grid
`{ceil_div(N, block_n), list_size-1}` where the second dimension encodes the number of
active experts.  Changing to a 1-D persistent scheduler would require restructuring the
kernel's block-to-work mapping.  The GPU builder approach (§4) is therefore the preferred
near-term fix; the persistent-scheduler approach is a longer-term architectural option that
would bring AsymGEMM's design closer to DeepGEMM's.

---

## 9. Reference Implementations

| Project | CUDA graph technique | Key file(s) |
|---------|---------------------|-------------|
| **DeepGEMM** | 1-D persistent scheduler; `masked_m` passed as GPU pointer; grid = `{num_sms, 1}` (constant) | `csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp`, `deep_gemm/include/deep_gemm/scheduler/gemm.cuh` |
| **Flash-Attention** | Side-stream warmup + `CUDAGraph` capture; `graph_pool_handle()` for memory sharing; in-place `copy_()` for input updates | `flash_attn/utils/generation.py` (`capture_graph`, `update_graph_cache`) |
| **AsymGEMM (this plan)** | Dense fixed-size GPU-built offsets/experts; early-exit guard; constant `grid_y = num_groups` | `csrc/indexing/main.cu`, GEMM impl `.cuh` files |
