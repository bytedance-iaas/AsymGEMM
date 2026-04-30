# SM80 MoE GEMM — Per-Expert SM Parallelism Plan

## Background

### SM100 approach (reference: `sm100_fp8_asym_gemm_1d1d.cuh`)

The SM100 kernel launches a **2-D grid**:

```
gridDim = (ceil_div(N, BLOCK_N),   // x: N-tile index
           num_groups)              // y: expert index
```

`blockIdx.y` directly selects the expert. The `asymScheduler` reads
`offsets[blockIdx.y*2]` and `offsets[blockIdx.y*2+1]` to obtain the
token range `[m_start, m_end)` for that expert. There is **no expert
loop inside the kernel** — every SM handles exactly one expert's work
for its assigned N-tile. Different experts run in parallel on different
SMs.

### SM80 current approach (`sm80_moe_gemm.cuh`)

Both kernels (`sm80_moe_gemm_impl` and `sm80_moe_fp8_gemm_impl`)
launch a **1-D grid**:

```
gridDim = (ceil_div(N, BLOCK_N),   // x: N-tile index
           1)                       // y: always 0, unused
```

Inside each CTA there is a serial loop over all experts:

```cpp
for (int e = 0; e < params.list_size; ++e) {
    const int32_t expert_id = params.expert_list[e];
    const int64_t len_start = (computed from previous index_list[e-1]);
    const int64_t len       = params.index_list[e] - len_start;
    // ... M-tile loop, K-tile loop
}
```

This serialises expert computation: all experts are processed one after
another by the same CTA. With many experts the GPU is under-utilised
because only `ceil_div(N, BLOCK_N)` CTAs are live at once.

---

## Goal

Adopt the SM100 pattern for SM80: assign **one expert per CTA** along
the grid-Y axis, so different experts run in parallel on different SMs.

---

## Files to Change

| File | Role |
|------|------|
| `asym_gemm/include/asym_gemm/impls/sm80_moe_gemm.cuh` | Device kernel: remove expert loop, derive expert from `blockIdx.y` |
| `csrc/jit_kernels/impls/sm80_moe_gemm.hpp` | Host launcher: change grid-Y from `1` to `list_size` |

`csrc/jit_kernels/heuristics/sm80.hpp` requires **no change** — it only
computes `grid_x`.

---

## Detailed Changes

### 1. `sm80_moe_gemm.cuh` — `sm80_moe_gemm_impl`

#### Current grid comment (lines 37-39)

```cpp
// Grid:  (ceil_div(N, BLOCK_N), 1)
//   blockIdx.x = which N-tile this CTA handles
//   blockIdx.y = 0 (unused; serial expert loop inside the kernel)
```

#### Replace with

```cpp
// Grid:  (ceil_div(N, BLOCK_N), list_size)
//   blockIdx.x = which N-tile this CTA handles
//   blockIdx.y = which expert entry (index into expert_list / index_list)
```

#### Current expert loop (lines 166-322)

```cpp
int64_t len_start = 0;

for (int e = 0; e < params.list_size; ++e) {
    const int32_t expert_id = params.expert_list[e];
    const int64_t len       = params.index_list[e] - len_start;
    // ... all work ...
    len_start = params.index_list[e];
}  // Expert loop
```

#### Replace with (no loop — derive from blockIdx.y)

```cpp
// One CTA = one expert.  blockIdx.y selects the expert entry.
const int     expert_e  = static_cast<int>(blockIdx.y);
const int32_t expert_id = params.expert_list[expert_e];
const int64_t len_start = (expert_e == 0)
                        ? 0LL
                        : static_cast<int64_t>(params.index_list[expert_e - 1]);
const int64_t len       = static_cast<int64_t>(params.index_list[expert_e]) - len_start;

// All code inside the old expert loop body stays unchanged from here:
// typed pointers, mX / mW / mO tensors, gX / gW / gO tiles, M-tile loop …
```

Everything **inside** the old loop body (the typed pointer casts, tensor
construction, `gX`/`gW`/`gO` tiling, M-tile loop, K-tile loop, output
write) is unchanged. Only the outer for-loop wrapper is removed and its
loop variables are replaced by the two lines above.

---

### 2. `sm80_moe_gemm.cuh` — `sm80_moe_fp8_gemm_impl`

Same transformation as above.

#### Current grid comment (lines 336-341)

```cpp
// Grid:  (ceil_div(N, BLOCK_N), 1)   — same as the BF16 kernel
// Block: (NWARPS * 32, 1, 1)
```

#### Replace with

```cpp
// Grid:  (ceil_div(N, BLOCK_N), list_size)
// Block: (NWARPS * 32, 1, 1)
```

#### Current expert loop (lines 487-678)

```cpp
int64_t len_start = 0;

for (int e = 0; e < params.list_size; ++e) {
    const int32_t expert_id = params.expert_list[e];
    const int64_t len       = params.index_list[e] - len_start;
    const int     m_max     = static_cast<int>((len + BLOCK_M - 1) / BLOCK_M);
    const int     k_max     = static_cast<int>(K / BLOCK_K);
    // ... k=0 path, k>0 path ...
    len_start = params.index_list[e];
}  // expert loop
```

#### Replace with

```cpp
const int     expert_e  = static_cast<int>(blockIdx.y);
const int32_t expert_id = params.expert_list[expert_e];
const int64_t len_start = (expert_e == 0)
                        ? 0LL
                        : static_cast<int64_t>(params.index_list[expert_e - 1]);
const int64_t len       = static_cast<int64_t>(params.index_list[expert_e]) - len_start;
const int     m_max     = static_cast<int>((len + BLOCK_M - 1) / BLOCK_M);
const int     k_max     = static_cast<int>(K / BLOCK_K);

// All code for k=0 path and k>0 path is unchanged from here.
```

---

### 3. `csrc/jit_kernels/impls/sm80_moe_gemm.hpp` — BF16/FP16 launcher

#### Current `LaunchArgs` (line 88)

```cpp
.launch_args = LaunchArgs({cfg.grid_x(static_cast<int>(N)), 1},
                           cfg.num_threads(),
                           cfg.smem_bytes()),
```

#### Replace with

```cpp
.launch_args = LaunchArgs({cfg.grid_x(static_cast<int>(N)), list_size},
                           cfg.num_threads(),
                           cfg.smem_bytes()),
```

`list_size` is already available as a parameter of
`sm80_m_grouped_moe_gemm_contiguous`.

---

### 4. `csrc/jit_kernels/impls/sm80_moe_gemm.hpp` — FP8 launcher

#### Current `LaunchArgs` (line 167-169)

```cpp
.launch_args = LaunchArgs({cfg.grid_x(static_cast<int>(N)), 1},
                           cfg.num_threads(),
                           smem_bytes),
```

#### Replace with

```cpp
.launch_args = LaunchArgs({cfg.grid_x(static_cast<int>(N)), list_size},
                           cfg.num_threads(),
                           smem_bytes),
```

`list_size` is already available as a parameter of
`sm80_m_grouped_fp8_moe_gemm_contiguous`.

---

## Invariants That Must Be Preserved

1. **`len_start` derivation is correct.**  
   `index_list[e]` stores cumulative end-token counts.  
   `len_start = index_list[expert_e - 1]` for `expert_e > 0`, and `0`
   for `expert_e == 0`. This is identical to what the serial loop
   accumulates.

2. **Empty-expert guard.**  
   If `len == 0` for a particular expert entry (e.g., no tokens routed
   to that expert), the CTA should exit early before doing any work.  
   Add at the top of the expert body (after computing `len`):

   ```cpp
   if (len == 0) return;
   ```

   This mirrors the SM100 early-exit at line 222-224 of
   `sm100_fp8_asym_gemm_1d1d.cuh`:
   ```cpp
   if constexpr (kGemmType == GemmType::MGroupedMasked) {
       if (scheduler.m_end == 0) return;
   }
   ```

3. **No shared-state between experts.**  
   The current serial loop accumulates `len_start` across iterations;
   removing the loop eliminates that shared state entirely. Each CTA
   is now fully independent.

4. **`blockIdx.y` is in `[0, list_size)`.**  
   The grid is launched with `gridDim.y = list_size`, so `blockIdx.y`
   is always a valid index into `expert_list` and `index_list`.

5. **Output pointer arithmetic unchanged.**  
   `o_e = o_g + len_start * N` — `len_start` now comes from
   `index_list[expert_e - 1]` instead of an accumulated variable, but
   the value is the same.

---

## Parallelism Impact

| | Current | After change |
|---|---|---|
| Grid Y | 1 | `list_size` (e.g., up to 256) |
| Expert processing | Serial inside CTA | Parallel across SMs |
| SMs active (N=7168, BLOCK_N=128, list_size=8) | 56 | 448 |
| SMs active (N=7168, BLOCK_N=128, list_size=64) | 56 | 3584 (wave-based) |

For typical MoE configurations (8–128 active experts) this change
turns the expert axis from dead serial work into additional
parallelism, matching the SM100 design philosophy.

---

## Testing Checklist

After implementation:

1. **Correctness**: run existing numeric tests
   (`asym_gemm/testing/numeric.py`) with the BF16, FP16, and FP8 MoE
   paths for multiple `list_size` values (1, 2, 8, 64).
2. **Edge case — single expert** (`list_size=1`): grid reduces to
   `(N_tiles, 1)`, same total CTAs as before.
3. **Edge case — zero-token expert**: verify the early `return` guard
   prevents out-of-bounds access when `len == 0`.
4. **Benchmark**: run `asym_gemm/testing/bench.py` to confirm
   throughput improves for `list_size > 1`.
