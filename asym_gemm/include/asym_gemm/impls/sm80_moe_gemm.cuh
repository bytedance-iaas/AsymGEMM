// asym_gemm/include/asym_gemm/impls/sm80_moe_gemm.cuh
#pragma once

#include <cstdint>
#include <type_traits>

#include <cutlass/numeric_types.h>   // cutlass::half_t, cutlass::bfloat16_t

#include <cute/tensor.hpp>
#include <cute/arch/mma_sm80.hpp>   // SM80_16x8x16_F32F16F16F32_TN, ...BF16...
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
    assert(K % static_cast<int64_t>(BLOCK_K) == 0 && "K must be divisible by BLOCK_K");
    assert(N % static_cast<int64_t>(BLOCK_N) == 0 && "N must be divisible by BLOCK_N");
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
                    Tensor tXgX_k = tXgX_m(_, _, k);
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
                    Tensor tXgX_k = tXgX_m(_, _, k);
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

}  // namespace asym_gemm
