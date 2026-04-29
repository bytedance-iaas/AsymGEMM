# SM89 FP8 MoE GEMM — Implementation Plan

**Target hardware:** RTX 4090 (SM89, Ada Lovelace)
**Target API:** `m_grouped_fp8_asym_gemm_sm80`

SM89 has **native FP8 tensor-core MMA** via `SM89_16x8x32_E4M3E4M3F32_TN`.
FP8 data stays as FP8 from source (HBM or CPU pinned) through smem and into MMA registers.
The only conversion is FP32→BF16 at the very end of each M-tile (accumulator → output).

---

## Memory layout assumptions

| Tensor | Storage | Data type |
|--------|---------|-----------|
| X (activations) | **HBM** | FP8 E4M3 |
| W (expert weights) | **CPU pinned memory** (PCIe-accessible) | FP8 E4M3 |
| O (output / partial sums) | **HBM** | BF16 |

Expert weight matrices reside in CPU pinned memory and are fetched across PCIe by
`cp.async`. To amortize the high latency of each PCIe transfer, each W K-tile is
loaded **once** before sweeping over all M-tiles — the K-outer, M-inner loop
structure from `mixtureExpertKernel.cu`.

---

## Data-flow

```
CPU pinned memory (FP8, 1 byte/elem)    HBM (FP8, 1 byte/elem)
    W[expert, N_tile, k]                    X[token_start:end, k]
          │  cp.async 128-bit=16 FP8              │  cp.async 128-bit=16 FP8
          ▼                                        ▼
      sW (FP8 smem)                          sX (FP8 smem)
          │  LDSM  SM75_U32x4_LDSM_N             │  LDSM  SM75_U32x4_LDSM_N
          ▼                                        ▼
    tOrW (FP8 registers)                  tSrX (FP8 registers)
                   │                                │
                   └──────── SM89 FP8 MMA ──────────┘
                                   │
                   FP32 accumulator (tSrO) — register-only, never touches HBM
                   seed from sO on k > 0: BF16 sO → FP32 tSrO (register convert)
                                   │  at final k only: × (scale_a * scale_b)
                                   │  FP32 → BF16  (register convert)
                                   ▼
                         sO (BF16 smem staging)
                                   │  gmem store  [BF16, 2 B/elem]
                                   ▼
                   HBM O[token_start:end, N_tile]  ◄── partial-sum read-back also BF16
```

**No FP8→BF16 conversion before MMA. No FP32 in HBM.**

### Why the accumulator is FP32 but HBM traffic is BF16

`SM89_16x8x32_E4M3E4M3F32_TN` mandates a **FP32 accumulator** at the hardware
level — there is no SM89 FP8 MMA variant with a BF16 or FP16 accumulator, and
even BF16×BF16 MMA on SM80/SM89 accumulates in FP32. The FP32 accumulator lives
only in registers throughout the kernel.

All HBM I/O uses BF16 (2 B/elem):

| HBM operation | Type | Size |
|---------------|------|------|
| Read X tile | FP8 | 1 B/elem |
| Read W tile (CPU pinned, via PCIe) | FP8 | 1 B/elem |
| Write partial-sum O (each M-tile) | **BF16** | 2 B/elem |
| Read partial-sum O seed (k > 0) | **BF16** | 2 B/elem |
| Final output write | **BF16** | 2 B/elem |

The FP32↔BF16 conversions (FP32 tSrO → BF16 rO before write; BF16 sO → FP32
tSrO seed on read-back) are pure register operations consuming no HBM bandwidth.
Storing BF16 partial sums rather than FP32 already halves the partial-sum
bandwidth compared to a naïve FP32-in-HBM approach.

---

## Why SM89 can skip the FP8→BF16 conversion

| Arch | FP8 MMA instruction | Conversion needed? |
|------|--------------------|--------------------|
| SM80 (A100) | None — BF16/FP16 only | Must convert FP8→BF16 before MMA |
| **SM89 (RTX 4090)** | `SM89_16x8x32_E4M3E4M3F32_TN` | **No — FP8 directly into MMA** |
| SM90 (H100) | WGMMA FP8 | No |

The MMA atom `SM89_16x8x32_E4M3E4M3F32_TN` accepts `float_e4m3_t` operands
directly and accumulates into `float` (FP32). LDSM loads 16 bytes = 16 FP8 per
thread, which exactly matches the A-fragment per thread for a 16×32 K-atom over
32 threads (16×32 / 32 = 16 elements).

---

## Kernel loop structure (K-outer, M-inner)

This follows the `cpuAwareTilingWithoutBias` pattern in `mixtureExpertKernel.cu`.
One CUDA thread block handles one expert and one N-tile (same grid as the BF16 kernel).
The block iterates K-tiles in the outer loop (amortising the CPU→GPU W load), and
M-tiles in the inner loop.

### k = 0 path

```
load W[expert, n_tile, k=0] from CPU pinned memory into sW  ← one load shared by all M-tiles
cp_async_fence(); cp_async_wait<0>(); __syncthreads();

for m = 0 .. m_max-1:
    clear(tSrO)                                              ← zero the FP32 accumulator
    load X[m, k=0] from HBM into sX  (predicated on M boundary)
    cp_async_fence(); cp_async_wait<0>(); __syncthreads();
    LDSM sX → tSrX,  LDSM sW → tOrW                        ← FP8 register load
    cute::gemm(tiled_mma, tSrO, tSrX, tOrW, tSrO)           ← FP8 MMA → FP32 tSrO
    if K == BLOCK_K (single tile): tSrO *= scale_a * scale_b ← scale only at final tile
    FP32 tSrO → BF16 rO
    rO → sO (smem_thr_copy_O.retile_S + copy)
    __syncthreads();
    sO → O[m] in HBM  (predicated on M boundary)
    __syncthreads();
```

### k > 0 path

```
for k = 1 .. k_max-1:
    load W[expert, n_tile, k] from CPU into sW               ← one load per K-tile
    cp_async_fence(); cp_async_wait<0>(); __syncthreads();

    for m = 0 .. m_max-1:
        clear(tSrO)                                          ← will be overwritten by seed
        load X[m, k] from HBM into sX   (predicated on M boundary)
        load O[m]    from HBM into sO   (predicated on M boundary)  ← BF16 partial sum seed
        cp_async_fence(); cp_async_wait<0>(); __syncthreads();
        seed tSrO from sO:                                   ← BF16 sO → FP32 tSrO
            tSrO_copy_view = smem_thr_copy_O.retile_D(tSrO)
            cute::copy(smem_tiled_copy_O, tSsO, tSrO_copy_view)
        LDSM sX → tSrX,  LDSM sW → tOrW
        cute::gemm(tiled_mma, tSrO, tSrX, tOrW, tSrO)       ← accumulates into seeded tSrO
        if k == k_max-1: tSrO *= scale_a * scale_b           ← scale only at final tile
        FP32 tSrO → BF16 rO
        rO → sO (smem_thr_copy_O.retile_S + copy)
        __syncthreads();
        sO → O[m] in HBM  (predicated)
        __syncthreads();
```

**Scale application rule:** `scale_a * scale_b` is applied to the FP32 accumulator
**only at the final K-tile** (`k == k_max-1` or `k_max == 1`). Intermediate writes
store unscaled BF16 partial sums so that future K-tiles can seed from an accurate
(unscaled) value.

---

## Seeding tSrO from BF16 sO (k > 0)

Mirrors lines 283-284 of `mixtureExpertKernel.cu`:

```cpp
Tensor tSrO_copy_view = smem_thr_copy_O.retile_D(tSrO);  // retile accumulator as copy dest
cute::copy(smem_tiled_copy_O, tSsO, tSrO_copy_view);      // BF16 sO → FP32 tSrO
```

`smem_thr_copy_O` is the C-side copy tile (`make_tiled_copy_C`) with the
`SmemCopyAtomO` atom. In our kernel, `SmemCopyAtomO` must be typed as
`Copy_Atom<UniversalCopy<uint32_t>, cutlass::bfloat16_t>` (not the FP8 atom)
so that the BF16→FP32 widening conversion happens when loading into the FP32
accumulator fragment.

---

## Scale factor application

Because FP8 operands enter the MMA unchanged, scales cannot be folded into the
smem values. Scale is applied to the FP32 accumulator at the **final K-tile write only**:

```cpp
const bool is_last_k = (k == k_max - 1);
if (is_last_k) {
    const float combined_scale = params.scale_a * params.scale_b;
    CUTE_UNROLL
    for (int i = 0; i < size(tSrO); i++)
        tSrO(i) *= combined_scale;
}
```

Then the FP32 accumulator is converted to BF16 for output:

```cpp
Tensor rO = make_tensor<cutlass::bfloat16_t>(shape(tSrO));
CUTE_UNROLL
for (int i = 0; i < size(tSrO); i++)
    rO(i) = cutlass::bfloat16_t(static_cast<float>(tSrO(i)));
```

---

## Smem layout and budget

### FP8 smem atom

For BF16 the existing atom is `Layout<Shape<_8, _64>, Stride<_64, _1>>` (8×64×2B = 1024B).
For FP8 (1 byte/elem) we double the column count to keep the same 1024-byte atom:

```cpp
using SmemLayoutAtomFP8 = decltype(composition(
    Swizzle<3, 3, 3>{},
    Layout<Shape<_8, _128>, Stride<_128, _1>>{}));  // 8×128×1B = 1024 B
```

This preserves the conflict-free LDSM access pattern: 32 threads each issuing a
16-byte load spanning 128 consecutive bytes exactly covers 32 banks once.

### BLOCK_K constraint

The SM89 FP8 MMA K-atom is **32** (vs 16 for BF16). `BLOCK_K` must be a multiple
of 32, minimum 32.

### Smem budget on SM89 (96 KB)

```
sX: BLOCK_M × BLOCK_K × 1 B  (FP8)
sW: BLOCK_N × BLOCK_K × 1 B  (FP8)
sO: BLOCK_M × BLOCK_N × 2 B  (BF16 — used for both output staging AND partial-sum seed read-back)

Total = (BLOCK_M + BLOCK_N) × BLOCK_K + BLOCK_M × BLOCK_N × 2
```

For BLOCK_M = BLOCK_N = 128:

| BLOCK_K | smem total | SM89 (96 KB)? |
|---------|-----------|----------------|
| 64  | 256×64 + 32768 = 49152 B | ✓ |
| 128 | 256×128 + 32768 = 65536 B | ✓ |
| 256 | 256×256 + 32768 = 98304 B | ✓ (exactly 96 KB) |
| 512 | 256×512 + 32768 = 163840 B | ✗ (over SM89 limit) |

Max BLOCK_K on SM89 = **256**.

---

## File Map

| File | Action | What changes |
|------|--------|-------------|
| `asym_gemm/include/asym_gemm/impls/sm80_moe_params.h` | **MODIFY** | Add `SM80MoEFP8Params` (adds `scale_a`, `scale_b`) |
| `asym_gemm/include/asym_gemm/impls/sm80_moe_gemm.cuh` | **MODIFY** | Append `sm80_moe_fp8_gemm_impl` kernel with SM89 native FP8 MMA + K-outer M-inner loop |
| `csrc/jit_kernels/heuristics/sm80.hpp` | **MODIFY** | Add FP8 smem helper + `select_sm80_fp8_config` |
| `csrc/jit_kernels/impls/sm80_moe_gemm.hpp` | **MODIFY** | Add `SM80MoEFP8GemmRuntime` + `sm80_m_grouped_fp8_moe_gemm_contiguous` |
| `csrc/apis/gemm.hpp` | **MODIFY** | Add `m_grouped_fp8_asym_gemm_sm80` + pybind registration |
| `asym_gemm/__init__.py` | **MODIFY** | Export `m_grouped_fp8_asym_gemm_sm80` |
| `tests/test_sm80_moe.py` | **MODIFY** | Replace BF16/FP16 tests with FP8 correctness test |

---

## Task 1 — Extend params header (`sm80_moe_params.h`)

```c
// asym_gemm/include/asym_gemm/impls/sm80_moe_params.h
#pragma once
#include <stdint.h>

#ifdef __cplusplus
namespace asym_gemm {
#endif

/* Existing BF16/FP16 params — unchanged */
typedef struct {
    void*    x_ptr;
    void*    w_ptr;
    void*    o_ptr;
    int32_t* expert_list;
    int32_t* index_list;
    int32_t  list_size;
    int32_t  expert_size;
    int64_t  N;
    int64_t  K;
} SM80MoEParams;

/*
 * FP8 params for SM89 native FP8 MMA.
 * x_ptr: float8_e4m3fn in HBM (row-major [total_tokens, K])
 * w_ptr: float8_e4m3fn in CPU pinned memory (row-major [num_experts, N, K])
 * o_ptr: bfloat16 in HBM (row-major [total_tokens, N])
 *        doubles as partial-sum accumulator across K-tiles
 * scale_a/b: per-tensor float32 scales applied to FP32 accumulator at final K-tile only
 */
typedef struct {
    void*    x_ptr;
    void*    w_ptr;
    void*    o_ptr;
    int32_t* expert_list;
    int32_t* index_list;
    int32_t  list_size;
    int32_t  expert_size;
    int64_t  N;
    int64_t  K;
    float    scale_a;
    float    scale_b;
} SM80MoEFP8Params;

#ifdef __cplusplus
}  // namespace asym_gemm
#endif
```

---

## Task 2 — Update heuristics (`csrc/jit_kernels/heuristics/sm80.hpp`)

Add alongside the existing BF16 helpers (do not modify them):

```cpp
// ── FP8 smem formula (sX+sW in FP8, sO in BF16) ─────────────────────────────
inline int smem_bytes_fp8(uint32_t block_m, uint32_t block_n, uint32_t block_k) {
    // sX: block_m * block_k * 1 byte (FP8)
    // sW: block_n * block_k * 1 byte (FP8)
    // sO: block_m * block_n * 2 bytes (BF16 — output staging + partial-sum seed read-back)
    return static_cast<int>((block_m + block_n) * block_k + block_m * block_n * 2);
}

// Max BLOCK_K for FP8 kernel on a given arch
// smem(BLOCK_K) = (128 + 128) * BLOCK_K + 128 * 128 * 2 = 256 * BLOCK_K + 32768
// SM89 ( 96 KB = 98304 B): BLOCK_K ≤ (98304  - 32768) / 256 = 256
inline int max_block_k_fp8(int arch_major, int arch_minor) {
    return (smem_limit(arch_major, arch_minor) - 32768) / 256;
}

// Config selector for the FP8 kernel.
// Key difference from select_sm80_config:
//   - BLOCK_K min is 32 (SM89 FP8 MMA K-atom = 32)
//   - BLOCK_K max is up to 256 on SM89 (double the BF16 limit of 128)
//   - smem uses FP8 formula
inline SM80GemmConfig select_sm80_fp8_config(int arch_major, int arch_minor,
                                             int N, int K) {
    const int smem_cap = smem_limit(arch_major, arch_minor);

    // Start at arch max, halve until K is divisible, min = 32
    uint32_t block_k = static_cast<uint32_t>(max_block_k_fp8(arch_major, arch_minor));
    while (block_k > 32u && K % static_cast<int>(block_k) != 0)
        block_k /= 2u;

    // Largest block_n in {128, 64, 32} that divides N and fits in smem
    uint32_t block_n = 32u;
    for (uint32_t bn : {128u, 64u, 32u}) {
        if (N % static_cast<int>(bn) != 0) continue;
        if (smem_bytes_fp8(128u, bn, block_k) <= smem_cap) {
            block_n = bn;
            break;
        }
    }

    return SM80GemmConfig{128u, block_n, block_k, 4u};
}
```

---

## Task 3 — Add FP8 kernel to `sm80_moe_gemm.cuh`

Append after the existing `sm80_moe_gemm_impl`. The existing kernel is **not modified**.

### Additional include

```cpp
#include <cute/arch/mma_sm89.hpp>  // SM89_16x8x32_E4M3E4M3F32_TN
```

### Kernel signature

```cpp
template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t NWARPS>
__global__ void sm80_moe_fp8_gemm_impl(SM80MoEFP8Params params);
```

### Architecture guard

```cpp
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
```

### Type aliases

```cpp
using ElementIn  = cutlass::float_e4m3_t;   // FP8 E4M3 — smem and MMA operands
using ElementOut = cutlass::bfloat16_t;     // output and partial-sum type
```

### Static asserts

```cpp
static_assert(BLOCK_K >= 32,       "BLOCK_K must be >= 32 (SM89 FP8 MMA K-atom)");
static_assert(BLOCK_K % 32 == 0,   "BLOCK_K must be a multiple of 32");
static_assert(BLOCK_M % (NWARPS * 16) == 0, "BLOCK_M must be divisible by NWARPS*16");
```

### Typed pointer casts and scalars

```cpp
const ElementIn* __restrict__ x_g = reinterpret_cast<const ElementIn*>(params.x_ptr);
const ElementIn* __restrict__ w_g = reinterpret_cast<const ElementIn*>(params.w_ptr);
ElementOut*      __restrict__ o_g = reinterpret_cast<ElementOut*>(params.o_ptr);
const float scale_a = params.scale_a;
const float scale_b = params.scale_b;
const int   k_max   = static_cast<int>((params.K + BLOCK_K - 1) / BLOCK_K);
```

### Smem layout — FP8 atoms

```cpp
// FP8 smem: Swizzle<3,3,3> with 8×128 atom (8×128×1B = 1024 B)
// Matches LDSM access: 32 threads × 16 bytes = 512 bytes per warp,
// spanning 128 consecutive bytes per thread — conflict-free across 32 banks.
using SmemLayoutAtomFP8 = decltype(composition(
    Swizzle<3, 3, 3>{},
    Layout<Shape<_8, _128>, Stride<_128, _1>>{}));
using SmemLayoutX = decltype(tile_to_shape(
    SmemLayoutAtomFP8{}, Shape<Int<BLOCK_M>, Int<BLOCK_K>>{}));
using SmemLayoutW = decltype(tile_to_shape(
    SmemLayoutAtomFP8{}, Shape<Int<BLOCK_N>, Int<BLOCK_K>>{}));
// Output / partial-sum staging: BF16, simple row-major
using SmemLayoutO = Layout<
    Shape<Int<BLOCK_M>, Int<BLOCK_N>>,
    Stride<Int<BLOCK_N>, _1>>;

// smem pointers: sX and sW hold FP8, sO holds BF16
constexpr int SMEM_X_ELEMS = BLOCK_M * BLOCK_K;  // FP8, 1 byte each
constexpr int SMEM_W_ELEMS = BLOCK_N * BLOCK_K;  // FP8, 1 byte each

extern __shared__ char smem_[];
ElementIn*  smem_x = reinterpret_cast<ElementIn*>(smem_);
ElementIn*  smem_w = smem_x + SMEM_X_ELEMS;
ElementOut* smem_o = reinterpret_cast<ElementOut*>(smem_w + SMEM_W_ELEMS);

Tensor sX = make_tensor(make_smem_ptr(smem_x), SmemLayoutX{});
Tensor sW = make_tensor(make_smem_ptr(smem_w), SmemLayoutW{});
Tensor sO = make_tensor(make_smem_ptr(smem_o), SmemLayoutO{});
```

### MMA setup — native SM89 FP8

```cpp
// SM89_16x8x32_E4M3E4M3F32_TN: FP8 E4M3 inputs, FP32 accumulator
using MMA_Op   = SM89_16x8x32_E4M3E4M3F32_TN;
using TiledMma = TiledMMA<
    MMA_Atom<MMA_Op>,
    Layout<Shape<Int<NWARPS>, _1, _1>>,
    Tile<Int<16 * NWARPS>, _16, _32>>;   // K-atom = 32 for FP8

TiledMma tiled_mma;
auto thr_mma = tiled_mma.get_thread_slice(tidx);

// FP32 accumulator
Tensor tSrO  = thr_mma.partition_fragment_C(sO);   // (MMA_C, MMA_M, MMA_N) — FP32
// FP8 MMA operand fragments — loaded directly from FP8 smem via LDSM
Tensor tSrX  = thr_mma.partition_fragment_A(sX);   // (MMA_A, MMA_M, MMA_K) — FP8
Tensor tOrW  = thr_mma.partition_fragment_B(sW);   // (MMA_B, MMA_N, MMA_K) — FP8
```

### LDSM setup — FP8 element type

```cpp
// SM75_U32x4_LDSM_N loads 16 bytes = 16 FP8 per thread.
// For A-fragment of SM89 16×8×32 over 32 threads: 16×32/32 = 16 elements. ✓
using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, ElementIn>;
auto smem_copy_A = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
auto smem_copy_B = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
auto smem_thr_copy_A = smem_copy_A.get_thread_slice(tidx);
auto smem_thr_copy_B = smem_copy_B.get_thread_slice(tidx);
Tensor tSsX = smem_thr_copy_A.partition_S(sX);
Tensor tOsW = smem_thr_copy_B.partition_S(sW);
```

### C-side smem copy — for seeding tSrO from sO (k > 0) and writing rO to sO

```cpp
// BF16 copy atom for sO ↔ accumulator fragment
using SmemCopyAtomO = Copy_Atom<UniversalCopy<uint32_t>, ElementOut>;
auto smem_tiled_copy_O   = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
auto smem_thr_copy_O     = smem_tiled_copy_O.get_thread_slice(tidx);
Tensor tSsO = smem_thr_copy_O.partition_S(sO);   // source for seed read (k>0)
```

### Global→smem copy atoms

```cpp
// FP8 copy for X (HBM) and W (CPU pinned): 128-bit = 16 FP8 per thread
using GmemCopyAtomFP8 = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, ElementIn>;
using GmemTiledCopyXW = decltype(make_tiled_copy(
    GmemCopyAtomFP8{},
    Layout<Shape<_32, _4>, Stride<_4, _1>>{},  // 128-thread layout
    Layout<Shape<_1, _16>>{}));                  // 16 FP8 per atom

// BF16 copy for O (HBM read-back in k>0 and write-out): 128-bit = 8 BF16 per thread
using GmemCopyAtomBF16 = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, ElementOut>;
using GmemTiledCopyO   = decltype(make_tiled_copy(
    GmemCopyAtomBF16{},
    Layout<Shape<_32, _4>, Stride<_4, _1>>{},
    Layout<Shape<_1, _8>>{}));                   // 8 BF16 per atom

GmemTiledCopyXW gmem_tiled_copy_xw;
GmemTiledCopyO  gmem_tiled_copy_o;
auto gmem_thr_copy_xw = gmem_tiled_copy_xw.get_thread_slice(tidx);
auto gmem_thr_copy_o  = gmem_tiled_copy_o.get_thread_slice(tidx);
```

### K-outer, M-inner loop structure

```cpp
int len_start = 0;
for (int e = 0; e < params.list_size; e++) {
    const int expert_id = params.expert_list[e];
    const int len       = params.index_list[e] - len_start;
    const int m_max     = (len + BLOCK_M - 1) / BLOCK_M;

    // Build gmem tensors for this expert
    Tensor mW = make_tensor(make_gmem_ptr(w_g + expert_id * params.N * params.K),
                            make_shape(Int<BLOCK_N>{}, params.K),
                            make_stride(params.K, _1{}));
    // ... gW, gX, gO local_tile tensors (same pattern as BF16 kernel) ...

    // ── k = 0: load W once, sweep M ──────────────────────────────────────
    // load W[n_tile, k=0] from CPU pinned memory → sW
    copy(gmem_tiled_copy_xw, tWgW(_, _, _, 0), tWsW);
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    for (int m = 0; m < m_max; m++) {
        clear(tSrO);

        // load X[m, k=0] from HBM → sX  (predicated on M boundary)
        // ... predicated copy identical to BF16 kernel ...
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        // LDSM sX → tSrX,  sW → tOrW
        Tensor tSrX_view = smem_thr_copy_A.retile_D(tSrX);
        cute::copy(smem_copy_A, tSsX, tSrX_view);
        Tensor tOrW_view = smem_thr_copy_B.retile_D(tOrW);
        cute::copy(smem_copy_B, tOsW, tOrW_view);

        // Native FP8 MMA
        cute::gemm(tiled_mma, tSrO, tSrX, tOrW, tSrO);

        // Apply scale only if this is also the final K-tile
        if (k_max == 1) {
            const float cs = scale_a * scale_b;
            CUTE_UNROLL
            for (int i = 0; i < size(tSrO); i++) tSrO(i) *= cs;
        }

        // FP32 → BF16 → sO → HBM
        Tensor rO = make_tensor<ElementOut>(shape(tSrO));
        CUTE_UNROLL
        for (int i = 0; i < size(tSrO); i++)
            rO(i) = ElementOut(static_cast<float>(tSrO(i)));
        Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
        Tensor taccOsO = smem_thr_copy_O.partition_D(sO);
        cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
        __syncthreads();
        // gmem store: sO → O[m] in HBM  (predicated on M boundary)
        // ...
        __syncthreads();
    }

    // ── k > 0: load W once per K-tile, sweep M reading back partial O ──
    for (int k = 1; k < k_max; k++) {
        // load W[n_tile, k] from CPU pinned memory → sW
        copy(gmem_tiled_copy_xw, tWgW(_, _, _, k), tWsW);
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        for (int m = 0; m < m_max; m++) {
            clear(tSrO);  // will be overwritten by seed

            // load X[m, k] from HBM → sX
            // load O[m]    from HBM → sO  (BF16 partial-sum seed)
            //   (both predicated on M boundary as in mixtureExpertKernel.cu)
            cp_async_fence();
            cp_async_wait<0>();
            __syncthreads();

            // Seed FP32 tSrO from BF16 sO (mirrors lines 283-284 of mixtureExpertKernel.cu)
            Tensor tSrO_copy_view = smem_thr_copy_O.retile_D(tSrO);
            cute::copy(smem_tiled_copy_O, tSsO, tSrO_copy_view);

            // LDSM sX → tSrX,  sW → tOrW
            Tensor tSrX_view = smem_thr_copy_A.retile_D(tSrX);
            cute::copy(smem_copy_A, tSsX, tSrX_view);
            Tensor tOrW_view = smem_thr_copy_B.retile_D(tOrW);
            cute::copy(smem_copy_B, tOsW, tOrW_view);

            // Native FP8 MMA — accumulates on top of seeded tSrO
            cute::gemm(tiled_mma, tSrO, tSrX, tOrW, tSrO);

            // Apply scale only at final K-tile
            if (k == k_max - 1) {
                const float cs = scale_a * scale_b;
                CUTE_UNROLL
                for (int i = 0; i < size(tSrO); i++) tSrO(i) *= cs;
            }

            // FP32 → BF16 → sO → HBM
            Tensor rO = make_tensor<ElementOut>(shape(tSrO));
            CUTE_UNROLL
            for (int i = 0; i < size(tSrO); i++)
                rO(i) = ElementOut(static_cast<float>(tSrO(i)));
            Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
            Tensor taccOsO = smem_thr_copy_O.partition_D(sO);
            cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
            __syncthreads();
            // gmem store: sO → O[m] in HBM  (predicated)
            // ...
            __syncthreads();
        }
    }

    len_start = params.index_list[e];
}
```

For `clear_smem_region` in the partial M-tile path, use `sizeof(ElementIn) = 1`:

```cpp
clear_smem_region<BLOCK_M * BLOCK_K * sizeof(ElementIn)>(
    reinterpret_cast<char*>(smem_x), tidx, NWARPS * 32);
```

---

## Task 4 — Add FP8 JIT runtime to `sm80_moe_gemm.hpp`

```cpp
class SM80MoEFP8GemmRuntime final : public LaunchRuntime<SM80MoEFP8GemmRuntime> {
public:
    struct Args {
        sm80::SM80GemmConfig gemm_config;
        LaunchArgs           launch_args;
        SM80MoEFP8Params     params;
    };

    static std::string generate_impl(const Args& args) {
        const auto& c = args.gemm_config;
        return fmt::format(R"(
#include <asym_gemm/impls/sm80_moe_gemm.cuh>
using namespace asym_gemm;
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &sm80_moe_fp8_gemm_impl<{}, {}, {}, {}>);
}};
)",
            c.block_m, c.block_n, c.block_k, c.nwarps);
    }

    static void launch_impl(const KernelHandle& kernel,
                            const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config, args.params));
    }
};

static void sm80_m_grouped_fp8_moe_gemm_contiguous(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& d,
    const torch::Tensor& expert_list,
    const torch::Tensor& index_list,
    int64_t N, int64_t K,
    int32_t num_experts, int32_t list_size,
    float scale_a, float scale_b)
{
    const auto& [arch_major, arch_minor] = device_runtime->get_arch_pair();
    const auto cfg = sm80::select_sm80_fp8_config(arch_major, arch_minor,
                                                   static_cast<int>(N),
                                                   static_cast<int>(K));

    const SM80MoEFP8Params params {
        .x_ptr       = a.data_ptr(),
        .w_ptr       = b.data_ptr(),
        .o_ptr       = d.data_ptr(),
        .expert_list = expert_list.data_ptr<int32_t>(),
        .index_list  = index_list.data_ptr<int32_t>(),
        .list_size   = list_size,
        .expert_size = num_experts,
        .N           = N,
        .K           = K,
        .scale_a     = scale_a,
        .scale_b     = scale_b,
    };

    const int smem_bytes = sm80::smem_bytes_fp8(cfg.block_m, cfg.block_n, cfg.block_k);

    const SM80MoEFP8GemmRuntime::Args runtime_args {
        .gemm_config = cfg,
        .launch_args = LaunchArgs({cfg.grid_x(static_cast<int>(N)), 1},
                                  cfg.num_threads(),
                                  smem_bytes),
        .params      = params,
    };

    const std::string kernel_name = fmt::format("sm80_moe_fp8_gemm_bm{}_bn{}_bk{}",
        cfg.block_m, cfg.block_n, cfg.block_k);

    const auto& code    = SM80MoEFP8GemmRuntime::generate(runtime_args);
    const auto& runtime = compiler->build(kernel_name, code);
    SM80MoEFP8GemmRuntime::launch(runtime, runtime_args);
}
```

---

## Task 5 — Wire API in `csrc/apis/gemm.hpp`

```cpp
static void m_grouped_fp8_asym_gemm_sm80(
    const torch::Tensor& a,        // [total_tokens, K]    float8_e4m3fn  (HBM)
    const torch::Tensor& b,        // [num_experts, N, K]  float8_e4m3fn  (CPU pinned or HBM)
    const torch::Tensor& d,        // [total_tokens, N]    bfloat16       (HBM)
    const torch::Tensor& offsets,  // [list_size] int32 cumulative end indices
    const torch::Tensor& experts,  // [list_size] int32 expert IDs
    const int&           list_size,
    const float&         scale_a,
    const float&         scale_b)
{
    DG_HOST_ASSERT(a.dim() == 2 && b.dim() == 3 && d.dim() == 2);

    const int64_t total_tokens = a.size(0);
    const int64_t K            = a.size(1);
    const int64_t num_experts  = b.size(0);
    const int64_t N            = b.size(1);
    DG_HOST_ASSERT(b.size(2) == K);
    DG_HOST_ASSERT(d.size(0) == total_tokens && d.size(1) == N);

    DG_HOST_ASSERT(a.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(b.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    DG_HOST_ASSERT(a.is_contiguous() && b.is_contiguous() && d.is_contiguous());
    DG_HOST_ASSERT(a.is_cuda() && d.is_cuda());
    // b may be on CPU pinned memory or CUDA device

    DG_HOST_ASSERT(offsets.is_cuda() && experts.is_cuda());
    DG_HOST_ASSERT(offsets.is_contiguous() && experts.is_contiguous());
    DG_HOST_ASSERT(offsets.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(experts.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(offsets.numel() >= list_size && experts.numel() >= list_size);

    if (total_tokens == 0 || N == 0 || K == 0) return;
    DG_HOST_ASSERT(K % 32 == 0 && K >= 32);  // FP8 MMA K-atom = 32
    DG_HOST_ASSERT(N % 32 == 0);

    sm80_m_grouped_fp8_moe_gemm_contiguous(
        a, b, d, experts, offsets,
        N, K,
        static_cast<int32_t>(num_experts),
        static_cast<int32_t>(list_size),
        scale_a, scale_b);
}
```

Register in `register_apis`:

```cpp
m.def("m_grouped_fp8_asym_gemm_sm80",
      static_cast<void(*)(const torch::Tensor&, const torch::Tensor&,
                          const torch::Tensor&, const torch::Tensor&,
                          const torch::Tensor&, const int&,
                          const float&, const float&)>(
          &m_grouped_fp8_asym_gemm_sm80),
      py::arg("a"), py::arg("b"), py::arg("d"),
      py::arg("offsets"), py::arg("experts"), py::arg("list_size"),
      py::arg("scale_a") = 1.0f,
      py::arg("scale_b") = 1.0f);
```

---

## Task 6 — Export in `asym_gemm/__init__.py`

```python
# SM80 MoE GEMM (BF16/FP16 and FP8-SM89, JIT)
"m_grouped_moe_gemm_nt_contiguous",
"m_grouped_fp8_asym_gemm_sm80",
```

---

## Task 7 — Update `tests/test_sm80_moe.py`

Replace the file with an FP8 correctness test for `m_grouped_fp8_asym_gemm_sm80`.

### Reference implementation

```python
@torch.no_grad()
def ref_fp8_moe_gemm(a_fp8, b_fp8, scale_a, scale_b, experts, offsets):
    """
    Dequantize FP8 inputs and compute MoE matmul in float32.
    Matches kernel semantics:  d = (a_fp8 @ b_fp8.T) * scale_a * scale_b

    a_fp8:   [total_tokens, K]    torch.float8_e4m3fn
    b_fp8:   [num_experts, N, K]  torch.float8_e4m3fn
    scale_a: float
    scale_b: float
    experts: int32 Tensor [list_size] — expert IDs
    offsets: int32 Tensor [list_size] — cumulative end-token indices
    returns: [total_tokens, N]    torch.float32
    """
    a_f32 = a_fp8.float()
    b_f32 = b_fp8.float()
    combined = scale_a * scale_b

    total_tokens = a_f32.shape[0]
    num_experts, N, K = b_f32.shape
    out = torch.zeros(total_tokens, N, dtype=torch.float32, device=a_fp8.device)

    elist = experts.tolist()
    ilist = offsets.tolist()
    start = 0
    for i, expert_id in enumerate(elist):
        end = ilist[i]
        out[start:end] = (a_f32[start:end] @ b_f32[expert_id].t()) * combined
        start = end
    return out
```

### Test cases

All satisfy the SM89 FP8 constraint `K % 32 == 0`:

```python
# (num_experts, N, K, token_counts_per_active_expert)
TEST_CASES = [
    (8,  4096,   256, [12, 8, 20, 28]),
    (8,  4096,   512, [128, 64, 256, 12]),
    (8,  4096,  4096, [256, 128, 512, 64, 300, 100, 200, 400]),
    (8,  4096,  7168, [128, 256, 64, 192]),
    (8,  4096,   128, [100, 200, 50, 150]),   # K=128 — 4 K-tiles at BLOCK_K=32
    (8, 16384,  4096, [64, 128, 32, 96]),
    (4,  4096,   256, [7, 13, 31, 127]),       # partial M-tiles
    (4,  4096,  4096, [512]),
]
```

### Test function

```python
import gc, itertools
import torch
import asym_gemm


def calc_diff(a, b):
    scale = b.abs().max().clamp(min=1e-6)
    return ((a - b) / scale).norm() / (a.numel() ** 0.5)


@torch.no_grad()
def test_fp8_moe_gemm():
    print('Testing m_grouped_fp8_asym_gemm_sm80 (SM89 native FP8 MMA):')
    all_passed = True
    scale_a, scale_b = 0.5, 2.0  # non-trivial scales to verify scale application

    for (num_experts, N, K, token_counts) in TEST_CASES:
        torch.cuda.empty_cache(); gc.collect()

        expert_ids   = list(range(len(token_counts)))
        total_tokens = sum(token_counts)
        offsets_h    = list(itertools.accumulate(token_counts))

        # Random BF16 in [-1, 1], cast to FP8 E4M3
        a_bf16 = torch.randn(total_tokens, K,    dtype=torch.bfloat16, device='cuda').clamp(-1, 1)
        b_bf16 = torch.randn(num_experts,  N, K, dtype=torch.bfloat16, device='cuda').clamp(-1, 1)
        a = a_bf16.to(torch.float8_e4m3fn)
        b = b_bf16.to(torch.float8_e4m3fn)
        d = torch.empty(total_tokens, N, dtype=torch.bfloat16, device='cuda')

        experts = torch.tensor(expert_ids, dtype=torch.int32, device='cuda')
        offsets = torch.tensor(offsets_h,  dtype=torch.int32, device='cuda')

        asym_gemm.m_grouped_fp8_asym_gemm_sm80(
            a, b, d, offsets, experts, len(expert_ids),
            scale_a=scale_a, scale_b=scale_b)
        torch.cuda.synchronize()

        ref = ref_fp8_moe_gemm(a, b, scale_a, scale_b, experts, offsets)

        diff = calc_diff(d.float(), ref)
        threshold = 0.01
        status = 'PASS' if diff < threshold else 'FAIL'
        print(f'[{status}]  N={N:5d} K={K:5d} tokens={total_tokens:5d} '
              f'experts={len(token_counts)}  diff={diff:.6f}')
        if diff >= threshold:
            all_passed = False

    return all_passed


if __name__ == '__main__':
    torch.manual_seed(42)
    ok = test_fp8_moe_gemm()
    if ok:
        print('\nAll tests passed.')
    else:
        raise SystemExit('One or more tests FAILED.')
```

---

## Correctness checklist

- [ ] Architecture guard is `__CUDA_ARCH__ >= 890` (not 800) — SM89 FP8 MMA instruction
- [ ] `SM89_16x8x32_E4M3E4M3F32_TN` included via `cute/arch/mma_sm89.hpp`
- [ ] `TiledMma` Tile K-dim is `_32` (not `_16`) — matches the FP8 MMA K-atom
- [ ] `SmemLayoutAtomFP8` uses `Layout<Shape<_8, _128>, Stride<_128, _1>>` (128 FP8 columns)
- [ ] `GmemTiledCopyXW` value layout is `Layout<Shape<_1, _16>>` (16 FP8 per 128-bit tx)
- [ ] `GmemTiledCopyO` value layout is `Layout<Shape<_1, _8>>` (8 BF16 per 128-bit tx)
- [ ] W is loaded **once per K-tile** before the M-loop (K-outer, M-inner pattern)
- [ ] k=0 path: `clear(tSrO)` → load X → MMA → (if k_max==1: apply scale) → write BF16 to HBM
- [ ] k>0 path: load W once → per-M: load X + load O from HBM → seed tSrO from sO → MMA → (if final k: apply scale) → write BF16 to HBM
- [ ] Seeding uses `smem_thr_copy_O.retile_D(tSrO)` + `cute::copy(smem_tiled_copy_O, tSsO, …)` — exact pattern from mixtureExpertKernel.cu lines 283-284
- [ ] Scale applied **only at final K-tile** (`k == k_max-1` or `k_max == 1`) — never to intermediate partial sums
- [ ] `clear_smem_region` byte count uses `sizeof(ElementIn) = 1` for sX (FP8)
- [ ] `smem_bytes_fp8(...)` passed to `LaunchArgs`, not `cfg.smem_bytes()` (BF16 formula)
- [ ] API assertion `DG_HOST_ASSERT(a.is_cuda() && d.is_cuda())` but NOT `b.is_cuda()` — b may be CPU pinned
- [ ] BLOCK_K % 32 == 0 enforced in both static assert (kernel) and runtime check (API)
- [ ] Reference: `(a.float() @ b.float().t()) * scale_a * scale_b` — scale applied to full result
- [ ] Test uses non-trivial scales (`scale_a=0.5, scale_b=2.0`) to confirm scale application
- [ ] Test inputs clamped to `[-1, 1]` before FP8 cast to avoid saturating E4M3 range (max ≈ 448)

---

## Smem verification

```
SM89 smem limit = 96 KB = 98304 B

BLOCK_M=128, BLOCK_N=128, BLOCK_K=256:
  sX = 128 × 256 × 1 = 32768 B
  sW = 128 × 256 × 1 = 32768 B
  sO = 128 × 128 × 2 = 32768 B
  Total = 98304 B = 96 KB ✓  (exactly at limit)

BLOCK_M=128, BLOCK_N=128, BLOCK_K=128:
  Total = 256×128 + 32768 = 65536 B = 64 KB ✓

BLOCK_M=128, BLOCK_N=128, BLOCK_K=64:
  Total = 256×64 + 32768 = 49152 B = 48 KB ✓
```

---

## Out of scope

- Per-block / per-group scale factors
- Pipelined double-buffered cp.async
- FP8 E5M2 variant
- Masked-layout variant
