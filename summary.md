# SM80 MoE GEMM Integration — Summary

## Goal

Add `asym_gemm.m_grouped_moe_gemm_nt_contiguous(x, w, o, expert_list, index_list)` to the existing `asym_gemm` Python package. The function dispatches on tensor dtype (FP16 or BF16) at runtime and runs a CuTe-native grouped GEMM kernel targeting SM80+ (A100, RTX 4090, GB200).

---

## Key Architecture Decisions

### 1. DeepGEMM JIT Pattern (not AOT compilation)

The existing package uses a JIT pattern:
- C++ `register_apis()` registers Python-callable functions via pybind11
- When called, the function selects block dimensions, fills a params struct, generates C++ source code as a string, and calls `compiler->build(kernel_name, code)`
- NVCC compiles the generated source into a CUBIN, cached at `~/.asym_gemm/cache/`
- Subsequent calls with the same `kernel_name` hit the CUBIN cache directly

The kernel `.cuh` file lives in `asym_gemm/include/` (the JIT compiler's include path). It is **not compiled at build time** — only at first runtime call.

### 2. Shared Plain-C Params Struct

`SM80MoEParams` is defined in a plain-C header (`sm80_moe_params.h`) includable from both the host C++ compiler and the NVCC JIT compiler:

```c
typedef struct {
    void*    x_ptr;       // [total_tokens, K] row-major
    void*    w_ptr;       // [num_experts, N, K] row-major
    void*    o_ptr;       // [total_tokens, N] row-major
    int32_t* expert_list; // [list_size] expert IDs
    int32_t* index_list;  // [list_size] cumulative end-token indices
    int32_t  list_size;
    int32_t  expert_size;
    int64_t  N;
    int64_t  K;
} SM80MoEParams;
```

`void*` data pointers make the struct layout-stable across compilation boundaries. The kernel casts to `Element*` internally.

### 3. JIT Cache Key Encodes Block Configuration

The kernel name (= CUBIN cache key) includes dtype and tile dimensions:
```
sm80_moe_gemm_fp16_bm128_bn128_bk256
sm80_moe_gemm_bf16_bm128_bn64_bk256
```

Different N or K values that lead to different tile configs get separate CUBINs. Without encoding the tile dims, two calls with different shapes could silently share a wrong CUBIN.

### 4. Block Size Heuristics

`select_sm80_config(arch_major, arch_minor, N, K)` in `sm80.hpp`:

- `BLOCK_M = 128` always (4 warps × 16-row MMA atom)
- `BLOCK_K`: start at arch max, halve until `K % BLOCK_K == 0`, min 64
  - SM80/SM100 (160 KB smem): max BLOCK_K = 256
  - SM89 (96 KB smem): max BLOCK_K = 128
- `BLOCK_N`: largest of {128, 64, 32} that divides N and fits in smem

Smem formula: `(BLOCK_M * BLOCK_K + BLOCK_N * BLOCK_K + BLOCK_M * BLOCK_N) * 2 bytes`

---

## Kernel Algorithm

**Grid**: `(ceil_div(N, BLOCK_N), 1)` — one CTA per N-tile, experts processed serially.

**Algorithm: M-outer, K-inner**

```
for each expert e in expert_list:
    x_slice = x[token_start : index_list[e]]   # [len, K]
    w_expert = w[expert_id]                     # [N, K]
    for each M-tile m:
        clear FP32 accumulator (tSrO)
        for each K-tile k:
            load sW[BLOCK_N, BLOCK_K] via cp.async
            load sX[BLOCK_M, BLOCK_K]:
                - full tile: cp.async (SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>)
                - partial last tile: zero sX cooperatively, predicated fill
            __syncthreads()
            LDSM sX → registers (SM75_U32x4_LDSM_N)
            LDSM sW → registers
            MMA: tSrO += tSrX @ tOrW^T
            __syncthreads()
        convert FP32 tSrO → Element → sO
        write sO → global (with M-row predicate for partial last tile)
```

FP32 accumulates across all K-tiles; writes back to global exactly once per M-tile. No global intermediate buffers.

### Shared Memory Layout

`Swizzle<3,3,3>` composed with an 8×64 base atom, extended via `tile_to_shape` to `BLOCK_M×BLOCK_K` and `BLOCK_N×BLOCK_K`. This eliminates bank conflicts for LDSM on FP16/BF16 data.

```cpp
using SmemLayoutAtom = decltype(composition(
    Swizzle<3, 3, 3>{},
    Layout<Shape<_8, _64>, Stride<_64, _1>>{}));
using SmemLayoutX = decltype(tile_to_shape(SmemLayoutAtom{},
    Shape<Int<BLOCK_M>, Int<BLOCK_K>>{}));
```

### MMA Atom Selection

```cpp
using MMA_Op = std::conditional_t<
    std::is_same_v<Element, cutlass::half_t>,
    SM80_16x8x16_F32F16F16F32_TN,
    SM80_16x8x16_F32BF16BF16F32_TN>;
```

Both atoms satisfy `__CUDA_ARCH__ >= 800` with no upper bound, so the kernel compiles and runs on SM89 and SM100 as well.

---

## Files Created / Modified

| File | Action | Purpose |
|------|--------|---------|
| `csrc/jit_kernels/heuristics/sm80.hpp` | CREATE | `SM80GemmConfig`, `select_sm80_config()` |
| `asym_gemm/include/asym_gemm/impls/sm80_moe_params.h` | CREATE | Shared plain-C `SM80MoEParams` |
| `csrc/jit_kernels/impls/sm80_moe_gemm.hpp` | CREATE | `SM80MoEGemmRuntime` CRTP + free function |
| `csrc/apis/gemm.hpp` | MODIFY | `m_grouped_moe_gemm_nt_contiguous()` + pybind11 registration |
| `asym_gemm/__init__.py` | MODIFY | Export `"m_grouped_moe_gemm_nt_contiguous"` |
| `tests/test_sm80_moe.py` | CREATE | 8 correctness test cases × 2 dtypes |
| `asym_gemm/include/asym_gemm/impls/sm80_moe_gemm.cuh` | CREATE | Complete CuTe-native kernel |

---

## Bugs Found and Fixed During Review

| Bug | Where | Fix |
|-----|-------|-----|
| Duplicate `SM80MoEParams` definition across host/device | Task 2 | Extract to shared plain-C header |
| `element_type_str.find("half")` matches `__half` | Task 2 | Exact equality `== "cutlass::half_t"` |
| Dead fields `n, k` in `SM80MoEGemmRuntime::Args` | Task 2 | Removed |
| Cache key missing block dims (different shapes share CUBIN) | Task 3 | Add `_bm{}_bn{}_bk{}` to key |
| Missing `x/w/o.is_cuda()` checks | Task 3 | Added before `data_ptr()` calls |
| Alignment assert `K >= 64` fires before empty-return when `K == 0` | Task 3 | Moved empty-return before alignment checks |
| `tXsX(_0{}, mi, ki)` skips elements within a copy atom | Task 6 | Full 3-loop `(ai, mi, ki)` iteration |
| Output predicate `ci % size<1>` fragile under layout changes | Task 6 | Explicit `(ai, mi, ni)` loop |
| No `K % BLOCK_K == 0` / `N % BLOCK_N == 0` enforcement in kernel | Task 6 | Added `assert()` at kernel entry |
| `tXgX_m(_, _, k)` — 3 coords for 4D tensor (compile error) | Task 7 | `tXgX_m(_, _, _, k)` |

---

## Test Results

All 16 test cases pass (8 shapes × FP16 + BF16):

```
[PASS] fp16  N= 4096 K=  256 tokens=   68 experts=4  diff=0.000042
[PASS] fp16  N= 4096 K=  512 tokens=  460 experts=4  diff=0.000040
[PASS] fp16  N= 4096 K= 4096 tokens= 1960 experts=8  diff=0.000040
[PASS] fp16  N= 4096 K= 7168 tokens=  640 experts=4  diff=0.000041
[PASS] fp16  N= 4096 K=  128 tokens=  500 experts=4  diff=0.000041
[PASS] fp16  N=16384 K= 4096 tokens=  320 experts=4  diff=0.000040
[PASS] fp16  N= 4096 K=  256 tokens=  178 experts=4  diff=0.000042
[PASS] fp16  N= 4096 K= 4096 tokens=  512 experts=1  diff=0.000037
[PASS] bf16  N= 4096 K=  256 tokens=   68 experts=4  diff=0.000331
[PASS] bf16  N= 4096 K=  512 tokens=  460 experts=4  diff=0.000320
[PASS] bf16  N= 4096 K= 4096 tokens= 1960 experts=8  diff=0.000303
[PASS] bf16  N= 4096 K= 7168 tokens=  640 experts=4  diff=0.000320
[PASS] bf16  N= 4096 K=  128 tokens=  500 experts=4  diff=0.000287
[PASS] bf16  N=16384 K= 4096 tokens=  320 experts=4  diff=0.000318
[PASS] bf16  N= 4096 K=  256 tokens=  178 experts=4  diff=0.000325
[PASS] bf16  N= 4096 K= 4096 tokens=  512 experts=1  diff=0.000339
```

Threshold: 0.01 (RMS relative error). Max observed: ~0.00034 (BF16).

---

## Usage

```python
import torch
import asym_gemm

# x: [total_tokens, K]          — fp16 or bf16, CUDA, contiguous
# w: [num_experts, N, K]        — same dtype as x, CUDA, contiguous
# o: [total_tokens, N]          — same dtype as x, CUDA, contiguous (output)
# expert_list: [list_size]      — int32, CUDA, expert IDs for each slot
# index_list:  [list_size]      — int32, CUDA, cumulative end-token index per slot

asym_gemm.m_grouped_moe_gemm_nt_contiguous(x, w, o, expert_list, index_list)
```

Constraints:
- `K >= 64`, `K % 16 == 0`, `N % 32 == 0`
- `expert_list` and `index_list` must have the same length
- First call JIT-compiles the kernel (~seconds); subsequent calls use the CUBIN cache

---

## Commit History

```
b69ea27 feat(sm80): complete SM80 MoE GEMM integration — all 16 tests passing
42135ad fix(sm80): fix 4D tensor indexing tXgX_m(_, _, _, k)
5917ec1 fix(sm80): guard divisibility asserts to thread 0
66e5b55 fix(sm80): add K/N divisibility asserts, rename NUM_ELEMS→NUM_BYTES
b7d6f62 fix(sm80): iterate all copy atom modes in partial-tile predicated copy
a83b00d feat(sm80): add CuTe-native SM80 MoE GEMM kernel header (FP16 + BF16)
0bc14c2 test(sm80): add correctness test (TDD)
6459972 fix(sm80): add CUDA placement checks, fix assertion ordering, fix cache key
9583343 feat(sm80): wire m_grouped_moe_gemm_nt_contiguous into Python API
52abd74 fix(sm80): add early element_type_str validation
461b5f2 fix(sm80): extract SM80MoEParams to shared header, fix kernel name check
13a10e1 feat(sm80): add SM80MoEGemmRuntime JIT wrapper
6362d6e fix(sm80): remove unused includes from heuristics header
f3dfe1f feat(sm80): add SM80GemmConfig and select_sm80_config heuristics
```
