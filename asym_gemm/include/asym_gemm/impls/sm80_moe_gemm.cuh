// asym_gemm/include/asym_gemm/impls/sm80_moe_gemm.cuh
#pragma once

#include <cstdint>
#include <type_traits>

#include <cutlass/numeric_types.h>   // cutlass::half_t, cutlass::bfloat16_t

#include <cute/tensor.hpp>
#include <cute/arch/mma_sm80.hpp>   // SM80_16x8x16_F32F16F16F32_TN, ...BF16...
#include <cute/arch/mma_sm89.hpp>   // SM89_16x8x32_F32E4M3E4M3F32_TN
#include <cute/arch/copy_sm80.hpp>  // SM80_CP_ASYNC_CACHEGLOBAL
#include <cute/arch/copy_sm75.hpp>  // SM75_U32x4_LDSM_N

#include <asym_gemm/impls/sm80_moe_params.h>  // SM80MoEParams

namespace asym_gemm {

using namespace cute;

// ──────────────────────────────────────────────────────────────────────────────
// Cooperative clear helper: zero a raw smem region using all threads
// ──────────────────────────────────────────────────────────────────────────────
template <int NUM_BYTES>
CUTE_DEVICE void clear_smem_region(char* ptr, int tidx, int num_threads) {
    static_assert(NUM_BYTES % 4 == 0, "NUM_BYTES must be 4-byte aligned");
    auto* iptr = reinterpret_cast<int32_t*>(ptr);
    constexpr int n_ints = NUM_BYTES / 4;
    CUTE_UNROLL
    for (int i = tidx; i < n_ints; i += num_threads)
        iptr[i] = 0;
}

// ──────────────────────────────────────────────────────────────────────────────
// Main kernel
//
// Grid:  (ceil_div(N, BLOCK_N), 1)
//   blockIdx.x = which N-tile this CTA handles
//   blockIdx.y = 0 (unused; serial expert loop inside the kernel)
// Block: (NWARPS * 32, 1, 1)
//
// Algorithm (M-outer, K-inner — accumulates in FP32, writes once per M-tile):
//   for each expert e in [0, list_size):
//     for each M-tile m:
//       clear FP32 accumulator
//       for each K-tile k:
//         load sX [BLOCK_M, BLOCK_K] from x[e][m][k]  (cp.async or element copy)
//         load sW [BLOCK_N, BLOCK_K] from w[expert_id][n_tile][k]  (cp.async)
//         fence + wait + syncthreads
//         LDSM sX → registers, LDSM sW → registers
//         MMA: acc += sX_reg @ sW_reg^T
//         syncthreads
//       convert acc (fp32) → Element, stage in sO, write sO → gO (with M predicate)
// ──────────────────────────────────────────────────────────────────────────────
template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t NWARPS, typename Element>
__global__ void sm80_moe_gemm_impl(SM80MoEParams params) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

    static_assert(BLOCK_K >= 64, "BLOCK_K must be >= 64 for the swizzled smem layout");
    static_assert(BLOCK_M % (NWARPS * 16) == 0, "BLOCK_M must be divisible by NWARPS*16");

    // ── Typed pointer casts ──────────────────────────────────────────────────
    // Runtime alignment assertions — the heuristic in select_sm80_config guarantees
    // these, but guard here so any future caller can see a clear failure site.
    const Element* __restrict__ x_g =
        reinterpret_cast<const Element*>(params.x_ptr);
    const Element* __restrict__ w_g =
        reinterpret_cast<const Element*>(params.w_ptr);
    Element* __restrict__ o_g =
        reinterpret_cast<Element*>(params.o_ptr);

    const int64_t N  = params.N;
    const int64_t K  = params.K;
    // Guard to thread 0 to avoid redundant per-thread abort floods on violation.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        assert(K % static_cast<int64_t>(BLOCK_K) == 0);
        assert(N % static_cast<int64_t>(BLOCK_N) == 0);
    }
    const int tidx   = static_cast<int>(threadIdx.x);
    const int n_tile = static_cast<int>(blockIdx.x);  // which N-block this CTA owns

    // ── Shared memory layout ─────────────────────────────────────────────────
    // Swizzle<3,3,3> with 8×64 base atom avoids bank conflicts for FP16/BF16 LDSM.
    // tile_to_shape extends it to BLOCK_M×BLOCK_K and BLOCK_N×BLOCK_K.
    using SmemLayoutAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        Layout<Shape<_8, _64>, Stride<_64, _1>>{}));
    using SmemLayoutX = decltype(tile_to_shape(
        SmemLayoutAtom{}, Shape<Int<BLOCK_M>, Int<BLOCK_K>>{}));
    using SmemLayoutW = decltype(tile_to_shape(
        SmemLayoutAtom{}, Shape<Int<BLOCK_N>, Int<BLOCK_K>>{}));
    // Output staging: simple row-major (no swizzle needed for non-MMA writes)
    using SmemLayoutO = Layout<
        Shape<Int<BLOCK_M>, Int<BLOCK_N>>,
        Stride<Int<BLOCK_N>, _1>>;

    // Smem offsets (in Elements)
    constexpr int SMEM_X_ELEMS = BLOCK_M * BLOCK_K;
    constexpr int SMEM_W_ELEMS = BLOCK_N * BLOCK_K;

    extern __shared__ char smem_[];
    Element* smem_base = reinterpret_cast<Element*>(smem_);
    Tensor sX = make_tensor(make_smem_ptr(smem_base),                    SmemLayoutX{});
    Tensor sW = make_tensor(make_smem_ptr(smem_base + SMEM_X_ELEMS),     SmemLayoutW{});
    Tensor sO = make_tensor(make_smem_ptr(smem_base + SMEM_X_ELEMS + SMEM_W_ELEMS), SmemLayoutO{});

    // ── MMA setup ────────────────────────────────────────────────────────────
    // Select FP16 or BF16 MMA atom at compile time
    using MMA_Op = std::conditional_t<
        std::is_same_v<Element, cutlass::half_t>,
        SM80_16x8x16_F32F16F16F32_TN,
        SM80_16x8x16_F32BF16BF16F32_TN>;

    // 4 warps in M direction; each warp executes one 16×8×16 atom.
    // Tile<16*NWARPS, 16, 16>: 64×16×16 per "tiled MMA call" for NWARPS=4.
    using TiledMma = TiledMMA<
        MMA_Atom<MMA_Op>,
        Layout<Shape<Int<NWARPS>, _1, _1>>,
        Tile<Int<16 * NWARPS>, _16, _16>>;

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);

    // Allocate accumulator and input register fragments
    Tensor tSrO = thr_mma.partition_fragment_C(sO);   // (MMA_C, MMA_M, MMA_N)  — FP32
    Tensor tSrX = thr_mma.partition_fragment_A(sX);   // (MMA_A, MMA_M, MMA_K)
    Tensor tOrW = thr_mma.partition_fragment_B(sW);   // (MMA_B, MMA_N, MMA_K)

    // ── Smem→register copy (LDSM) ────────────────────────────────────────────
    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    auto smem_copy_A = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
    auto smem_copy_B = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_A = smem_copy_A.get_thread_slice(tidx);
    auto smem_thr_copy_B = smem_copy_B.get_thread_slice(tidx);
    // Source partitions in smem for LDSM
    Tensor tSsX = smem_thr_copy_A.partition_S(sX);   // (LDSM_ATOM, LDSM_M, LDSM_K)
    Tensor tOsW = smem_thr_copy_B.partition_S(sW);   // (LDSM_ATOM, LDSM_N, LDSM_K)

    // ── Global→smem copy (cp.async, 128-bit) ─────────────────────────────────
    // Thread layout: 32 threads × 4 threads = 128 threads total.
    // Each thread copies 1×8 elements (8 FP16 = 128 bits) per transaction.
    using GmemCopyAtom   = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, Element>;
    using GmemTiledCopyXW = decltype(make_tiled_copy(
        GmemCopyAtom{},
        Layout<Shape<_32, _4>, Stride<_4, _1>>{},
        Layout<Shape<_1,  _8>>{}));
    GmemTiledCopyXW gmem_tiled_copy_xw;
    auto gmem_thr_copy_xw = gmem_tiled_copy_xw.get_thread_slice(tidx);

    // ── Output smem→gmem copy (synchronous, non-async) ───────────────────────
    using GmemCopyAtomO  = Copy_Atom<UniversalCopy<uint128_t>, Element>;
    using GmemTiledCopyO = decltype(make_tiled_copy(
        GmemCopyAtomO{},
        Layout<Shape<_32, _4>, Stride<_4, _1>>{},
        Layout<Shape<_1,  _8>>{}));
    GmemTiledCopyO gmem_tiled_copy_o;
    auto gmem_thr_copy_o = gmem_tiled_copy_o.get_thread_slice(tidx);

    // ── Smem→register copy for output staging ────────────────────────────────
    using SmemCopyAtomO  = Copy_Atom<UniversalCopy<Element>, Element>;
    auto smem_copy_O     = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_copy_O.get_thread_slice(tidx);

    // ── Expert loop ──────────────────────────────────────────────────────────
    int64_t len_start = 0;

    for (int e = 0; e < params.list_size; ++e) {
        const int32_t expert_id = params.expert_list[e];
        const int64_t len       = params.index_list[e] - len_start;

        // Typed pointers for this expert's slice
        const Element* x_e = x_g + len_start * K;                 // [len, K]
        const Element* w_e = w_g + (int64_t)expert_id * N * K;    // [N, K]
        Element*       o_e = o_g + len_start * N;                  // [len, N]

        // Global tensors for this expert
        // Note: mX has runtime M=len; we use local_tile to get (BLOCK_M,BLOCK_K,M_tiles,K_tiles)
        Tensor mX = make_tensor(make_gmem_ptr(x_e),
                                make_shape(len, K), make_stride(K, Int<1>{}));
        Tensor mW = make_tensor(make_gmem_ptr(w_e),
                                make_shape(N,   K), make_stride(K, Int<1>{}));
        Tensor mO = make_tensor(make_gmem_ptr(o_e),
                                make_shape(len, N), make_stride(N, Int<1>{}));

        // Tile into blocks
        // gX: (BLOCK_M, BLOCK_K, M_tiles, K_tiles)
        Tensor gX = local_tile(mX, Shape<Int<BLOCK_M>, Int<BLOCK_K>>{}, make_coord(_, _));
        // gW: (BLOCK_N, BLOCK_K, K_tiles)   — fixed to this CTA's n_tile
        Tensor gW = local_tile(mW, Shape<Int<BLOCK_N>, Int<BLOCK_K>>{}, make_coord(n_tile, _));
        // gO: (BLOCK_M, BLOCK_N, M_tiles)   — fixed to this CTA's n_tile
        Tensor gO = local_tile(mO, Shape<Int<BLOCK_M>, Int<BLOCK_N>>{}, make_coord(_, n_tile));

        const int M_tiles = static_cast<int>(size<2>(gX));
        const int K_tiles = static_cast<int>(size<3>(gX));  // = K / BLOCK_K

        // Coordinate tensor for M predication
        Tensor cX = make_identity_tensor(Shape<Int<BLOCK_M>, Int<BLOCK_K>>{});

        // Partition W once (same source for all m_tiles per k_tile)
        Tensor tWsW     = gmem_thr_copy_xw.partition_D(sW);
        Tensor tOsO_src = gmem_thr_copy_o.partition_S(sO);

        // ── M-tile loop ──────────────────────────────────────────────────────
        for (int m = 0; m < M_tiles; ++m) {
            // Actual rows in this M-tile (< BLOCK_M only for the last tile)
            const int m_actual = static_cast<int>(
                cute::min((int64_t)BLOCK_M, len - (int64_t)m * BLOCK_M));

            // gX for this m-tile: (BLOCK_M, BLOCK_K, K_tiles)
            Tensor gX_m = gX(_, _, m, _);

            // Partition X and its coordinate tensor
            Tensor tXsX     = gmem_thr_copy_xw.partition_D(sX);
            Tensor tXgX_m   = gmem_thr_copy_xw.partition_S(gX_m);  // (COPY, COPY_M, K_tiles)
            Tensor tXcX     = gmem_thr_copy_xw.partition_S(cX);    // (COPY, COPY_M, COPY_K)

            // Clear FP32 accumulator for this M-tile
            clear(tSrO);

            // ── K-tile loop ──────────────────────────────────────────────────
            for (int k = 0; k < K_tiles; ++k) {
                // Partition W for this k-tile
                Tensor tWgW_k = gmem_thr_copy_xw.partition_S(gW(_, _, k));

                // Load W tile (always full: N is a multiple of BLOCK_N by construction)
                cute::copy(gmem_tiled_copy_xw, tWgW_k, tWsW);
                cp_async_fence();

                // Load X tile — predicated if this is the last (partial) M-tile
                if (m_actual == static_cast<int>(BLOCK_M)) {
                    // Full tile: use async copy
                    Tensor tXgX_k = tXgX_m(_, _, _, k);
                    cute::copy(gmem_tiled_copy_xw, tXgX_k, tXsX);
                    cp_async_fence();
                    cp_async_wait<0>();
                } else {
                    // Partial tile: wait for W, then synchronous predicated copy
                    cp_async_wait<0>();
                    __syncthreads();

                    // Zero sX cooperatively, then fill valid rows
                    clear_smem_region<BLOCK_M * BLOCK_K * sizeof(Element)>(
                        reinterpret_cast<char*>(smem_base),
                        tidx, static_cast<int>(NWARPS * 32));
                    __syncthreads();

                    // Element-wise copy with M predicate — iterate all three modes
                    // (atom, M, K) so every element within a copy atom is handled.
                    Tensor tXgX_k = tXgX_m(_, _, _, k);
                    for (int mi = 0; mi < size<1>(tXsX); mi++) {
                        int m_coord = get<0>(tXcX(_0{}, mi, _0{}));
                        if (m_coord < m_actual) {
                            for (int ai = 0; ai < size<0>(tXsX); ai++) {
                                for (int ki = 0; ki < size<2>(tXsX); ki++) {
                                    tXsX(ai, mi, ki) = tXgX_k(ai, mi, ki);
                                }
                            }
                        }
                    }
                }

                __syncthreads();

                // ── LDSM: smem → registers ───────────────────────────────────
                Tensor tSrX_view = smem_thr_copy_A.retile_D(tSrX);
                cute::copy(smem_copy_A, tSsX, tSrX_view);

                Tensor tOrW_view = smem_thr_copy_B.retile_D(tOrW);
                cute::copy(smem_copy_B, tOsW, tOrW_view);

                // ── MMA: tSrO += tSrX @ tOrW^T ───────────────────────────────
                cute::gemm(tiled_mma, tSrO, tSrX, tOrW, tSrO);

                __syncthreads();
            }  // K-tile loop

            // ── Convert FP32 accumulator → Element ───────────────────────────
            Tensor rO = make_tensor<Element>(shape(tSrO));
            CUTE_UNROLL
            for (int i = 0; i < size(tSrO); i++)
                rO(i) = Element(static_cast<float>(tSrO(i)));

            // ── Stage output in smem ──────────────────────────────────────────
            Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
            Tensor taccOsO = smem_thr_copy_O.partition_D(sO);
            cute::copy(smem_copy_O, taccOrO, taccOsO);
            __syncthreads();

            // ── Write sO → gO (with M predicate for partial last tile) ────────
            // Safety invariant for partial tiles: MMA accumulates over zeroed sX rows
            // (via clear_smem_region + predicated fill), so accumulators for M rows
            // >= m_actual are 0.0. The M predicate below prevents writing those zeros
            // to global memory. tOrO is read from sO for all rows but only written for
            // valid rows — the stale sO values (zeros) for out-of-bounds rows are safe.
            Tensor gO_m  = gO(_, _, m);               // (BLOCK_M, BLOCK_N)
            Tensor cO    = make_identity_tensor(Shape<Int<BLOCK_M>, Int<BLOCK_N>>{});
            Tensor tOcO  = gmem_thr_copy_o.partition_S(cO);
            Tensor tOgO  = gmem_thr_copy_o.partition_D(gO_m);

            // Load sO → register buffer
            Tensor tOrO  = make_tensor<Element>(shape(tOgO));
            cute::copy(gmem_tiled_copy_o, tOsO_src, tOrO);

            // Write to global with M predicate — iterate modes (atom, M, N)
            // explicitly so the M-coordinate check is unambiguous.
            for (int mi = 0; mi < size<1>(tOgO); mi++) {
                int m_coord = get<0>(tOcO(_0{}, mi, _0{}));
                if (m_coord < m_actual) {
                    for (int ai = 0; ai < size<0>(tOgO); ai++) {
                        for (int ni = 0; ni < size<2>(tOgO); ni++) {
                            tOgO(ai, mi, ni) = tOrO(ai, mi, ni);
                        }
                    }
                }
            }

            __syncthreads();
        }  // M-tile loop

        len_start = params.index_list[e];
    }  // Expert loop

#else
    // Architecture guard: this kernel requires SM80+
    if (blockIdx.x == 0 && threadIdx.x == 0)
        printf("sm80_moe_gemm_impl: requires __CUDA_ARCH__ >= 800, got %d\n",
               static_cast<int>(__CUDA_ARCH__));
#endif
}

// ──────────────────────────────────────────────────────────────────────────────
// FP8 MoE GEMM kernel — SM89 (RTX 4090) native FP8 MMA
//
// Grid:  (ceil_div(N, BLOCK_N), 1)   — same as the BF16 kernel
// Block: (NWARPS * 32, 1, 1)
//
// Algorithm: K-outer, M-inner (mixtureExpertKernel.cu pattern)
//   W lives in CPU pinned memory — load once per K-tile to amortise PCIe latency.
//   O in HBM holds BF16 partial sums between K-tiles; FP32 accumulator is register-only.
//
// Smem atom: Swizzle<3,3,4> + 32×32 FP8 (= 1024 B, same footprint as 8×64 BF16).
//   S=4 ensures that 16 consecutive FP8 (= 16 bytes, one cp.async/LDSM unit) land
//   at contiguous smem addresses after the XOR.  The 32-col atom is the minimum
//   divisible by BLOCK_K (which is a multiple of 32).
// ──────────────────────────────────────────────────────────────────────────────
template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t NWARPS>
__global__ void sm80_moe_fp8_gemm_impl(SM80MoEFP8Params params) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890

    static_assert(BLOCK_K >= 32,       "BLOCK_K must be >= 32 (SM89 FP8 MMA K-atom)");
    static_assert(BLOCK_K % 32 == 0,   "BLOCK_K must be a multiple of 32");
    static_assert(BLOCK_M % (NWARPS * 16) == 0, "BLOCK_M must be divisible by NWARPS*16");

    using ElementIn  = cutlass::float_e4m3_t;   // FP8 E4M3 — smem and MMA operands
    using ElementOut = cutlass::bfloat16_t;     // BF16 — output and partial-sum HBM type

    const ElementIn*  __restrict__ x_g = reinterpret_cast<const ElementIn*>(params.x_ptr);
    const ElementIn*  __restrict__ w_g = reinterpret_cast<const ElementIn*>(params.w_ptr);
    ElementOut*       __restrict__ o_g = reinterpret_cast<ElementOut*>(params.o_ptr);

    const int64_t N      = params.N;
    const int64_t K      = params.K;
    const int     tidx   = static_cast<int>(threadIdx.x);
    const int     n_tile = static_cast<int>(blockIdx.x);

    // ── Shared memory layout ─────────────────────────────────────────────────
    // FP8 sX/sW: plain row-major (no swizzle).  CuTe's compile-time algebra
    // cannot prove stride-1 for a swizzled 16-element window when the smem
    // partition spans multiple M-atoms; the plain layout gives compile-time
    // strides throughout at the cost of bank conflicts (acceptable for
    // correctness; can be optimised later with a padded layout).
    using SmemLayoutX = Layout<Shape<Int<BLOCK_M>, Int<BLOCK_K>>,
                               Stride<Int<BLOCK_K>, _1>>;
    using SmemLayoutW = Layout<Shape<Int<BLOCK_N>, Int<BLOCK_K>>,
                               Stride<Int<BLOCK_K>, _1>>;
    // sO: BF16, row-major — output staging AND partial-sum read-back seed
    using SmemLayoutO = Layout<Shape<Int<BLOCK_M>, Int<BLOCK_N>>,
                               Stride<Int<BLOCK_N>, _1>>;

    constexpr int SMEM_X_ELEMS = BLOCK_M * BLOCK_K;   // FP8, 1 byte each
    constexpr int SMEM_W_ELEMS = BLOCK_N * BLOCK_K;   // FP8, 1 byte each

    extern __shared__ char smem_[];
    ElementIn*  smem_x = reinterpret_cast<ElementIn*>(smem_);
    ElementIn*  smem_w = smem_x + SMEM_X_ELEMS;
    ElementOut* smem_o = reinterpret_cast<ElementOut*>(smem_w + SMEM_W_ELEMS);

    Tensor sX = make_tensor(make_smem_ptr(smem_x), SmemLayoutX{});
    Tensor sW = make_tensor(make_smem_ptr(smem_w), SmemLayoutW{});
    Tensor sO = make_tensor(make_smem_ptr(smem_o), SmemLayoutO{});

    // ── MMA: SM89 native FP8, FP32 accumulator ──────────────────────────────
    using MMA_Op   = SM89_16x8x32_F32E4M3E4M3F32_TN;
    using TiledMma = TiledMMA<
        MMA_Atom<MMA_Op>,
        Layout<Shape<Int<NWARPS>, _1, _1>>,
        Tile<Int<16 * NWARPS>, _16, _32>>;   // K-atom = 32 for FP8

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);

    Tensor tSrO = thr_mma.partition_fragment_C(sO);   // FP32 accumulator
    Tensor tSrX = thr_mma.partition_fragment_A(sX);   // FP8 A-fragment
    Tensor tOrW = thr_mma.partition_fragment_B(sW);   // FP8 B-fragment

    // ── LDSM: FP8 smem → FP8 registers ─────────────────────────────────────
    // SM75_U32x4_LDSM_N loads 4×uint32 = 16 bytes = 16 FP8 per thread.
    // With plain row-major smem, each LDSM thread provides an 8-byte address
    // and the instruction does the warp-cooperative distribution; the two 8-
    // element groups within a thread's partition have compile-time inter-stride
    // Int<BLOCK_K>, so CuTe's vectorisation check passes.
    using SmemCopyAtomFP8 = Copy_Atom<SM75_U32x4_LDSM_N, ElementIn>;
    auto smem_copy_A      = make_tiled_copy_A(SmemCopyAtomFP8{}, tiled_mma);
    auto smem_copy_B      = make_tiled_copy_B(SmemCopyAtomFP8{}, tiled_mma);
    auto smem_thr_copy_A  = smem_copy_A.get_thread_slice(tidx);
    auto smem_thr_copy_B  = smem_copy_B.get_thread_slice(tidx);
    Tensor tSsX = smem_thr_copy_A.partition_S(sX);
    Tensor tOsW = smem_thr_copy_B.partition_S(sW);

    // ── C-side smem copy: sO (BF16) ↔ rO register buffer (BF16) ────────────
    // NOTE: tSrO is FP32 (mandated by SM89 FP8 MMA).  We cannot use retile_D
    // on tSrO with a BF16 copy atom — that would be a bitwise type mismatch.
    // Instead: copy BF16 sO → BF16 register buffer (rO_seed / rO), then convert
    // BF16↔FP32 in software.  SmemCopyAtomO is element-wise BF16.
    using SmemCopyAtomO   = Copy_Atom<UniversalCopy<ElementOut>, ElementOut>;
    auto smem_copy_O      = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O  = smem_copy_O.get_thread_slice(tidx);
    Tensor tSsO = smem_thr_copy_O.partition_S(sO);   // smem source for k>0 seed

    // ── Global→smem copies ───────────────────────────────────────────────────
    // FP8 (X and W): 128-bit tx = 16 FP8 per thread. Plain row-major smem → stride 1.
    using GmemCopyAtomFP8 = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, ElementIn>;
    using GmemTiledCopyXW = decltype(make_tiled_copy(
        GmemCopyAtomFP8{},
        Layout<Shape<_32, _4>, Stride<_4, _1>>{},
        Layout<Shape<_1, _16>>{}));                // 16 FP8 per atom
    GmemTiledCopyXW gmem_tiled_copy_xw;
    auto gmem_thr_copy_xw = gmem_tiled_copy_xw.get_thread_slice(tidx);

    // BF16 (O write-back and smem read): 128-bit tx = 8 BF16 per thread.
    // UniversalCopy dispatches to LDS.128 (smem→register) and STG (register→gmem),
    // so it works for both the sO→register read in write_output and element-wise
    // global accesses in the partial-tile O-seed path.
    using GmemCopyAtomBF16 = Copy_Atom<UniversalCopy<uint128_t>, ElementOut>;
    using GmemTiledCopyO   = decltype(make_tiled_copy(
        GmemCopyAtomBF16{},
        Layout<Shape<_32, _4>, Stride<_4, _1>>{},
        Layout<Shape<_1, _8>>{}));                 // 8 BF16 per atom
    GmemTiledCopyO gmem_tiled_copy_o;
    auto gmem_thr_copy_o = gmem_tiled_copy_o.get_thread_slice(tidx);

    // Coordinate tensors for M-predication
    Tensor cX = make_identity_tensor(Shape<Int<BLOCK_M>, Int<BLOCK_K>>{});
    Tensor cO = make_identity_tensor(Shape<Int<BLOCK_M>, Int<BLOCK_N>>{});

    // ── Helper: write BF16 rO register buffer → sO, then sO → gO ───────────
    // Extracted as a lambda to avoid code duplication between k=0 and k>0 paths.
    // rO: BF16 register tensor with same shape as tSrO.
    // gO_m: gmem slice for the current M-tile.
    // m_actual: number of valid M rows in this tile.
    auto write_output = [&](auto& rO, auto& gO_m, int m_actual) {
        // rO (BF16 registers) → sO (BF16 smem)
        Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
        Tensor taccOsO = smem_thr_copy_O.partition_D(sO);
        cute::copy(smem_copy_O, taccOrO, taccOsO);
        __syncthreads();

        // sO → gO_m (with M predicate for partial last tile)
        Tensor tOsO_src = gmem_thr_copy_o.partition_S(sO);
        Tensor tOcO     = gmem_thr_copy_o.partition_S(cO);
        Tensor tOgO     = gmem_thr_copy_o.partition_D(gO_m);
        Tensor tOrO     = make_tensor<ElementOut>(shape(tOgO));
        cute::copy(gmem_tiled_copy_o, tOsO_src, tOrO);
        for (int mi = 0; mi < size<1>(tOgO); mi++) {
            int m_coord = get<0>(tOcO(_0{}, mi, _0{}));
            if (m_coord < m_actual) {
                for (int ai = 0; ai < size<0>(tOgO); ai++)
                    for (int ni = 0; ni < size<2>(tOgO); ni++)
                        tOgO(ai, mi, ni) = tOrO(ai, mi, ni);
            }
        }
        __syncthreads();
    };

    // ── Expert loop ──────────────────────────────────────────────────────────
    int64_t len_start = 0;

    for (int e = 0; e < params.list_size; ++e) {
        const int32_t expert_id = params.expert_list[e];
        const int64_t len       = params.index_list[e] - len_start;
        const int     m_max     = static_cast<int>((len + BLOCK_M - 1) / BLOCK_M);
        const int     k_max     = static_cast<int>(K / BLOCK_K);

        const ElementIn* x_e = x_g + len_start * K;
        const ElementIn* w_e = w_g + (int64_t)expert_id * N * K;
        ElementOut*      o_e = o_g + len_start * N;

        // Global tensors for this expert
        Tensor mX = make_tensor(make_gmem_ptr(x_e),
                                make_shape(len, K), make_stride(K, Int<1>{}));
        Tensor mW = make_tensor(make_gmem_ptr(w_e),
                                make_shape(N,   K), make_stride(K, Int<1>{}));
        Tensor mO = make_tensor(make_gmem_ptr(o_e),
                                make_shape(len, N), make_stride(N, Int<1>{}));

        Tensor gX = local_tile(mX, Shape<Int<BLOCK_M>, Int<BLOCK_K>>{}, make_coord(_, _));
        Tensor gW = local_tile(mW, Shape<Int<BLOCK_N>, Int<BLOCK_K>>{}, make_coord(n_tile, _));
        Tensor gO = local_tile(mO, Shape<Int<BLOCK_M>, Int<BLOCK_N>>{}, make_coord(_, n_tile));

        Tensor tWsW = gmem_thr_copy_xw.partition_D(sW);   // W → sW destination

        // ── k = 0 path: load W once, sweep all M-tiles ───────────────────────
        {
            Tensor tWgW_0 = gmem_thr_copy_xw.partition_S(gW(_, _, 0));
            cute::copy(gmem_tiled_copy_xw, tWgW_0, tWsW);
            cp_async_fence();
            cp_async_wait<0>();
            __syncthreads();

            for (int m = 0; m < m_max; ++m) {
                const int m_actual = static_cast<int>(
                    cute::min((int64_t)BLOCK_M, len - (int64_t)m * BLOCK_M));

                clear(tSrO);

                // Load X[m, k=0] → sX
                Tensor gX_m    = gX(_, _, m, _);
                Tensor tXsX    = gmem_thr_copy_xw.partition_D(sX);
                Tensor tXgX_mk = gmem_thr_copy_xw.partition_S(gX_m(_, _, 0));
                Tensor tXcX    = gmem_thr_copy_xw.partition_S(cX);

                if (m_actual == static_cast<int>(BLOCK_M)) {
                    cute::copy(gmem_tiled_copy_xw, tXgX_mk, tXsX);
                    cp_async_fence();
                    cp_async_wait<0>();
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
                __syncthreads();

                // LDSM: sX → tSrX,  sW → tOrW
                Tensor tSrX_view = smem_thr_copy_A.retile_D(tSrX);
                cute::copy(smem_copy_A, tSsX, tSrX_view);
                Tensor tOrW_view = smem_thr_copy_B.retile_D(tOrW);
                cute::copy(smem_copy_B, tOsW, tOrW_view);

                // Native FP8 MMA → FP32 accumulator
                cute::gemm(tiled_mma, tSrO, tSrX, tOrW, tSrO);

                // Apply scale if this is the only K-tile
                if (k_max == 1) {
                    const float cs = params.scale_a * params.scale_b;
                    CUTE_UNROLL
                    for (int i = 0; i < size(tSrO); i++) tSrO(i) *= cs;
                }

                // FP32 → BF16 register buffer, then write to sO → gO
                Tensor rO = make_tensor<ElementOut>(shape(tSrO));
                CUTE_UNROLL
                for (int i = 0; i < size(tSrO); i++)
                    rO(i) = ElementOut(static_cast<float>(tSrO(i)));
                Tensor gO_m = gO(_, _, m);
                write_output(rO, gO_m, m_actual);
            }  // m-loop (k=0)
        }

        // ── k > 0 path: load W once per K-tile, per-M: seed tSrO from HBM O ─
        for (int k = 1; k < k_max; ++k) {
            Tensor tWgW_k = gmem_thr_copy_xw.partition_S(gW(_, _, k));
            cute::copy(gmem_tiled_copy_xw, tWgW_k, tWsW);
            cp_async_fence();
            cp_async_wait<0>();
            __syncthreads();

            const bool is_last_k = (k == k_max - 1);

            for (int m = 0; m < m_max; ++m) {
                const int m_actual = static_cast<int>(
                    cute::min((int64_t)BLOCK_M, len - (int64_t)m * BLOCK_M));

                // Load X[m,k] → sX
                Tensor gX_m    = gX(_, _, m, _);
                Tensor gO_m    = gO(_, _, m);
                Tensor tXsX    = gmem_thr_copy_xw.partition_D(sX);
                Tensor tXgX_mk = gmem_thr_copy_xw.partition_S(gX_m(_, _, k));
                Tensor tXcX    = gmem_thr_copy_xw.partition_S(cX);

                if (m_actual == static_cast<int>(BLOCK_M)) {
                    cute::copy(gmem_tiled_copy_xw, tXgX_mk, tXsX);
                    cp_async_fence();
                    cp_async_wait<0>();
                    // Seed FP32 tSrO directly from HBM O (all rows valid — no predicate).
                    // thr_mma.partition_C maps the MMA C-fragment thread assignment onto
                    // gO_m, giving each thread the BF16 partial sums it owns.
                    // Each tSgO(i) is a scalar global load (LDG); vectorisation happens
                    // naturally via the compiler since the accesses are affine.
                    Tensor tSgO = thr_mma.partition_C(gO_m);
                    CUTE_UNROLL
                    for (int i = 0; i < size(tSrO); i++)
                        tSrO(i) = static_cast<float>(tSgO(i));
                } else {
                    cp_async_wait<0>();
                    __syncthreads();
                    clear_smem_region<BLOCK_M * BLOCK_K * sizeof(ElementIn)>(
                        reinterpret_cast<char*>(smem_x), tidx, NWARPS * 32);
                    __syncthreads();
                    // Predicated X copy
                    for (int mi = 0; mi < size<1>(tXsX); mi++) {
                        int m_coord = get<0>(tXcX(_0{}, mi, _0{}));
                        if (m_coord < m_actual)
                            for (int ai = 0; ai < size<0>(tXsX); ai++)
                                for (int ki = 0; ki < size<2>(tXsX); ki++)
                                    tXsX(ai, mi, ki) = tXgX_mk(ai, mi, ki);
                    }
                    // Predicated O read-back into sO.
                    // Valid rows are loaded from HBM; invalid rows are zeroed in smem
                    // so that the sO→FP32 conversion below initialises those rows to 0.0.
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
                    // All threads have written their portion of sO; must sync before
                    // reading back with the MMA-C partition (different thread mapping).
                    __syncthreads();
                    // sO (BF16 smem) → BF16 register buffer → FP32 accumulator.
                    Tensor rO_seed      = make_tensor<ElementOut>(shape(tSrO));
                    Tensor rO_seed_view = smem_thr_copy_O.retile_D(rO_seed);
                    cute::copy(smem_copy_O, tSsO, rO_seed_view);
                    CUTE_UNROLL
                    for (int i = 0; i < size(tSrO); i++)
                        tSrO(i) = static_cast<float>(rO_seed(i));
                }

                // LDSM: sX → tSrX,  sW → tOrW
                Tensor tSrX_view = smem_thr_copy_A.retile_D(tSrX);
                cute::copy(smem_copy_A, tSsX, tSrX_view);
                Tensor tOrW_view = smem_thr_copy_B.retile_D(tOrW);
                cute::copy(smem_copy_B, tOsW, tOrW_view);

                // Native FP8 MMA — accumulates into seeded FP32 tSrO
                cute::gemm(tiled_mma, tSrO, tSrX, tOrW, tSrO);

                // Apply scale only at the final K-tile
                if (is_last_k) {
                    const float cs = params.scale_a * params.scale_b;
                    CUTE_UNROLL
                    for (int i = 0; i < size(tSrO); i++) tSrO(i) *= cs;
                }

                // FP32 → BF16 register buffer, then write to sO → gO
                Tensor rO = make_tensor<ElementOut>(shape(tSrO));
                CUTE_UNROLL
                for (int i = 0; i < size(tSrO); i++)
                    rO(i) = ElementOut(static_cast<float>(tSrO(i)));
                write_output(rO, gO_m, m_actual);
            }  // m-loop (k>0)
        }  // k-loop

        len_start = params.index_list[e];
    }  // expert loop

#else
    if (blockIdx.x == 0 && threadIdx.x == 0)
        printf("sm80_moe_fp8_gemm_impl: requires __CUDA_ARCH__ >= 890\n");
#endif
}

}  // namespace asym_gemm
