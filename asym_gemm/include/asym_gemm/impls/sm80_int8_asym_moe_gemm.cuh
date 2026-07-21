// asym_gemm/include/asym_gemm/impls/sm80_int8_asym_moe_gemm.cuh
// INT8 asym grouped MoE GEMM for SM80+ (A100 and later).
//
// SM80 counterpart of the SM90 1d1d INT8 asym kernel, structured like the
// SM89 FP8 kernel (sm89_fp8_moe_gemm.cuh): K-outer / M-inner so a weight tile
// streamed from CPU pinned memory over PCIe is loaded ONCE per K-tile and
// reused across every M-tile; FP32 partial sums live in HBM (o_ptr) between
// K-tiles. Differences from the FP8 kernel:
//
//   * mma.m16n8k32.s8 with S32 accumulation — exact within a K-tile, so all
//     the FP8 block-scale machinery (inter-group ratio folding, seed
//     reciprocal rescale) disappears. BLOCK_K is locked to 128 = the scale
//     granularity; each K-tile applies its (per-token x per-channel) scale
//     exactly once at the S32 -> F32 boundary.
//   * Partials are FP32 (matching the SM90 1d1d kernel's D convention), and
//     the epilogue is a direct element-wise RMW on global memory through the
//     MMA-C partition — no smem output staging. One CTA owns one
//     (n-block, expert) pair for ALL K-tiles, so the RMW is race-free.
//   * Scales are the natural layouts: sfa [total_tokens, kb], sfb [E, N, kb].
//
// This header must stay compilable with -arch=sm_80 (see the sm_80 gate in
// tests/test_arch_compile_gates.py).
#pragma once

#include <cstdint>

#include <cute/tensor.hpp>
#include <cute/arch/mma_sm80.hpp>   // SM80_16x8x32_S32S8S8S32_TN
#include <cute/arch/copy_sm80.hpp>  // SM80_CP_ASYNC_CACHEGLOBAL
#include <cute/arch/copy_sm75.hpp>  // SM75_U32x4_LDSM_N

#include <asym_gemm/impls/smxx_moe_utils.cuh>     // moe_predicated_copy
#include <asym_gemm/impls/sm80_int8_moe_params.h> // SM80MoEInt8Params, SM80MoEInt8MaskedParams

namespace asym_gemm {

using namespace cute;

// ──────────────────────────────────────────────────────────────────────────────
// Shared body: one CTA computes o[0:len, n_tile*BLOCK_N : +BLOCK_N] for one
// expert slice (x_e [len,K], w_e [N,K], o_e [len,N], sfa_e [len,kb],
// sfb_e [N,kb]).  Callers derive the slices from their own grouping scheme.
//
// Loop order (K-outer, M-inner — mixtureExpertKernel.cu pattern):
//   for each K-tile k (BLOCK_K = 128 = one scale k-group):
//     load sW [BLOCK_N, BLOCK_K] once (cp.async, possibly over PCIe)
//     stage sfb[:, k] (BLOCK_N floats) in smem, LDSM sW -> registers
//     for each M-tile m:
//       load sX [BLOCK_M, BLOCK_K] (cp.async), stage sfa[m-tile, k] in smem
//       LDSM sX, clear S32 accumulator, MMA over the 4 K-atoms
//       o[m,n] (+)= float(acc_s32) * sfa[m,k] * sfb[n,k]   (= on k==0)
// ──────────────────────────────────────────────────────────────────────────────
template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t NWARPS>
CUTE_DEVICE void sm80_int8_asym_moe_gemm_body(
    const int8_t* __restrict__ x_e,
    const int8_t* __restrict__ w_e,
    float*        __restrict__ o_e,
    const float*  __restrict__ sfa_e,
    const float*  __restrict__ sfb_e,
    int64_t len, int64_t N, int64_t K, int32_t kb,
    int n_tile, int tidx, char* smem_)
{
    static_assert(BLOCK_K == 128,
                  "BLOCK_K is locked to the 128-element scale granularity");
    static_assert(BLOCK_M % (NWARPS * 16) == 0,
                  "BLOCK_M must be divisible by NWARPS*16");
    static_assert(BLOCK_M <= NWARPS * 32,
                  "scale staging assumes one thread per BLOCK_M row");

    // ── Shared memory: row-major int8 tiles + per-tile scale vectors ────────
    using SmemLayoutX = Layout<Shape<Int<BLOCK_M>, Int<BLOCK_K>>,
                               Stride<Int<BLOCK_K>, _1>>;
    using SmemLayoutW = Layout<Shape<Int<BLOCK_N>, Int<BLOCK_K>>,
                               Stride<Int<BLOCK_K>, _1>>;

    constexpr int SMEM_X_BYTES = BLOCK_M * BLOCK_K;
    constexpr int SMEM_W_BYTES = BLOCK_N * BLOCK_K;

    int8_t* smem_x  = reinterpret_cast<int8_t*>(smem_);
    int8_t* smem_w  = smem_x + SMEM_X_BYTES;
    float*  smem_sa = reinterpret_cast<float*>(smem_ + SMEM_X_BYTES + SMEM_W_BYTES);
    float*  smem_sb = smem_sa + BLOCK_M;

    Tensor sX = make_tensor(make_smem_ptr(smem_x), SmemLayoutX{});
    Tensor sW = make_tensor(make_smem_ptr(smem_w), SmemLayoutW{});

    // ── MMA: SM80 native INT8, S32 accumulator ──────────────────────────────
    using MMA_Op   = SM80_16x8x32_S32S8S8S32_TN;
    using TiledMma = TiledMMA<
        MMA_Atom<MMA_Op>,
        Layout<Shape<Int<NWARPS>, _1, _1>>,
        Tile<Int<16 * NWARPS>, _16, _32>>;   // K-atom = 32 for INT8

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);

    Tensor tSrX = thr_mma.partition_fragment_A(sX);   // int8 A-fragment
    Tensor tOrW = thr_mma.partition_fragment_B(sW);   // int8 B-fragment

    // ── LDSM: int8 smem -> registers (16 B / thread, as in the FP8 kernel) ──
    using SmemCopyAtomS8 = Copy_Atom<SM75_U32x4_LDSM_N, int8_t>;
    auto smem_copy_A     = make_tiled_copy_A(SmemCopyAtomS8{}, tiled_mma);
    auto smem_copy_B     = make_tiled_copy_B(SmemCopyAtomS8{}, tiled_mma);
    auto smem_thr_copy_A = smem_copy_A.get_thread_slice(tidx);
    auto smem_thr_copy_B = smem_copy_B.get_thread_slice(tidx);
    Tensor tSsX = smem_thr_copy_A.partition_S(sX);
    Tensor tOsW = smem_thr_copy_B.partition_S(sW);

    // ── Global -> smem: 128-bit cp.async = 16 int8 per thread ───────────────
    using GmemCopyAtomS8  = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, int8_t>;
    using GmemTiledCopyXW = decltype(make_tiled_copy(
        GmemCopyAtomS8{},
        Layout<Shape<_32, _4>, Stride<_4, _1>>{},
        Layout<Shape<_1, _16>>{}));
    GmemTiledCopyXW gmem_tiled_copy_xw;
    auto gmem_thr_copy_xw = gmem_tiled_copy_xw.get_thread_slice(tidx);

    // Coordinate / predicate tensors for partial-M X loads
    Tensor cX   = make_identity_tensor(Shape<Int<BLOCK_M>, Int<BLOCK_K>>{});
    Tensor tXpX = make_tensor<bool>(make_shape(size<2>(gmem_thr_copy_xw.partition_D(sX))));
    cute::fill(tXpX, true);

    // MMA-C coordinates: per accumulator element, its (m, n) coord in-tile.
    Tensor cO   = make_identity_tensor(Shape<Int<BLOCK_M>, Int<BLOCK_N>>{});
    Tensor tScO = thr_mma.partition_C(cO);

    // Global tensors for this expert slice
    Tensor mX = make_tensor(make_gmem_ptr(x_e),
                            make_shape(len, K), make_stride(K, Int<1>{}));
    Tensor mW = make_tensor(make_gmem_ptr(w_e),
                            make_shape(N,   K), make_stride(K, Int<1>{}));
    Tensor mO = make_tensor(make_gmem_ptr(o_e),
                            make_shape(len, N), make_stride(N, Int<1>{}));

    Tensor gX = local_tile(mX, Shape<Int<BLOCK_M>, Int<BLOCK_K>>{}, make_coord(_, _));
    Tensor gW = local_tile(mW, Shape<Int<BLOCK_N>, Int<BLOCK_K>>{}, make_coord(n_tile, _));
    Tensor gO = local_tile(mO, Shape<Int<BLOCK_M>, Int<BLOCK_N>>{}, make_coord(_, n_tile));

    // S32 accumulator fragment (shape from the static C-tile partition)
    Tensor tSrAcc = thr_mma.partition_fragment_C(gO(_, _, 0));

    Tensor tWsW = gmem_thr_copy_xw.partition_D(sW);

    const int m_max = static_cast<int>((len + BLOCK_M - 1) / BLOCK_M);
    const int k_max = static_cast<int>(K / BLOCK_K);

    // sfb rows for this CTA's N-block: sfb_e[(n_tile*BLOCK_N + i) * kb + k]
    const float* sfb_n = sfb_e + (int64_t)n_tile * BLOCK_N * kb;

    // ── K-tile loop (outer: W loaded once per K-tile) ────────────────────────
    for (int k = 0; k < k_max; ++k) {
        // Load W tile + stage this K-block's per-channel scales.
        Tensor tWgW_k = gmem_thr_copy_xw.partition_S(gW(_, _, k));
        cute::copy(gmem_tiled_copy_xw, tWgW_k, tWsW);
        for (int i = tidx; i < static_cast<int>(BLOCK_N); i += NWARPS * 32)
            smem_sb[i] = sfb_n[(int64_t)i * kb + k];
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        // LDSM W once per K-tile; registers stay valid across the M-loop.
        Tensor tOrW_view = smem_thr_copy_B.retile_D(tOrW);
        cute::copy(smem_copy_B, tOsW, tOrW_view);

        // ── M-tile loop ──────────────────────────────────────────────────────
        for (int m = 0; m < m_max; ++m) {
            const int m_actual = static_cast<int>(
                cute::min((int64_t)BLOCK_M, len - (int64_t)m * BLOCK_M));

            // Load X tile + stage this M-tile's per-token scales.
            Tensor gX_m    = gX(_, _, m, _);
            Tensor tXsX    = gmem_thr_copy_xw.partition_D(sX);
            Tensor tXgX_mk = gmem_thr_copy_xw.partition_S(gX_m(_, _, k));
            Tensor tXcX    = gmem_thr_copy_xw.partition_S(cX);

            if (m_actual == static_cast<int>(BLOCK_M)) {
                cute::copy(gmem_tiled_copy_xw, tXgX_mk, tXsX);
            } else {
                moe_predicated_copy</*Is_even_MN=*/false, /*Is_even_K=*/true,
                                    /*Clear_OOB_MN=*/true,  /*Clear_OOB_K=*/true>(
                    gmem_tiled_copy_xw, tXgX_mk, tXsX, tXcX, tXpX, m_actual);
            }
            if (tidx < static_cast<int>(BLOCK_M)) {
                const int64_t m_global = (int64_t)m * BLOCK_M + tidx;
                smem_sa[tidx] = (m_global < len)
                    ? sfa_e[m_global * kb + k] : 0.0f;
            }
            cp_async_fence();
            cp_async_wait<0>();
            __syncthreads();

            // LDSM X, then INT8 MMA over the K-tile's 4 K-atoms.
            Tensor tSrX_view = smem_thr_copy_A.retile_D(tSrX);
            cute::copy(smem_copy_A, tSsX, tSrX_view);

            clear(tSrAcc);
            cute::gemm(tiled_mma, tSrAcc, tSrX, tOrW, tSrAcc);

            // ── Epilogue: exact S32 -> F32 dequant, RMW FP32 partials in HBM ─
            // Element-wise through the MMA-C partition; each element is owned
            // by the same thread for every K-tile, so the k>0 read observes
            // this thread's own k-1 write (same-address program order).
            Tensor gO_m = gO(_, _, m);
            Tensor tSgO = thr_mma.partition_C(gO_m);
            CUTE_UNROLL
            for (int i = 0; i < size(tSrAcc); ++i) {
                const int m_coord = get<0>(tScO(i));
                if (m_coord < m_actual) {
                    const float contrib = static_cast<float>(tSrAcc(i))
                                        * smem_sa[m_coord]
                                        * smem_sb[get<1>(tScO(i))];
                    tSgO(i) = (k == 0) ? contrib : (tSgO(i) + contrib);
                }
            }

            // Protect sX / smem_sa from the next iteration's overwrite.
            __syncthreads();
        }  // m-loop
    }  // k-loop
}

// ──────────────────────────────────────────────────────────────────────────────
// Contiguous grouped kernel
//
// Grid:  (ceil_div(N, BLOCK_N), list_size - 1)
// Block: (NWARPS * 32, 1, 1)
// blockIdx.y = segment index. Same segment convention as the SM90 1d1d asym
// kernel (asymScheduler.cuh): rows [index_list[2i], index_list[2i+1]) of the
// global token array, expert_list[i] indexes W/sfb, expert_id < 0 = skip.
// ──────────────────────────────────────────────────────────────────────────────
template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t NWARPS>
__global__ void sm80_int8_asym_moe_gemm_impl(SM80MoEInt8Params params) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

    extern __shared__ char smem_[];

    const int     expert_e  = static_cast<int>(blockIdx.y);
    const int32_t expert_id = params.expert_list[expert_e];
    const int64_t len_start = static_cast<int64_t>(params.index_list[2 * expert_e]);
    const int64_t len       = static_cast<int64_t>(params.index_list[2 * expert_e + 1])
                            - len_start;

    if (expert_id < 0 || len <= 0) return;

    const int64_t N  = params.N;
    const int64_t K  = params.K;
    const int32_t kb = params.kb;

    sm80_int8_asym_moe_gemm_body<BLOCK_M, BLOCK_N, BLOCK_K, NWARPS>(
        reinterpret_cast<const int8_t*>(params.x_ptr) + len_start * K,
        reinterpret_cast<const int8_t*>(params.w_ptr) + (int64_t)expert_id * N * K,
        reinterpret_cast<float*>(params.o_ptr) + len_start * N,
        params.sfa_ptr + len_start * kb,
        params.sfb_ptr + (int64_t)expert_id * N * kb,
        len, N, K, kb,
        static_cast<int>(blockIdx.x), static_cast<int>(threadIdx.x), smem_);

#else
    if (blockIdx.x == 0 && threadIdx.x == 0)
        printf("sm80_int8_asym_moe_gemm_impl: requires __CUDA_ARCH__ >= 800\n");
#endif
}

// ──────────────────────────────────────────────────────────────────────────────
// Masked kernel — padded [G, M_max, ...] layout, constant grid, graph-safe
//
// Grid:  (ceil_div(N, BLOCK_N), num_groups)
// Block: (NWARPS * 32, 1, 1)
// ──────────────────────────────────────────────────────────────────────────────
template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t NWARPS>
__global__ void sm80_int8_asym_moe_gemm_masked_impl(SM80MoEInt8MaskedParams params) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

    extern __shared__ char smem_[];

    const int     g   = static_cast<int>(blockIdx.y);
    const int64_t len = static_cast<int64_t>(params.masked_m[g]);
    if (len == 0) return;

    const int64_t M_max = params.M_max;
    const int64_t N     = params.N;
    const int64_t K     = params.K;
    const int32_t kb    = params.kb;

    sm80_int8_asym_moe_gemm_body<BLOCK_M, BLOCK_N, BLOCK_K, NWARPS>(
        reinterpret_cast<const int8_t*>(params.x_ptr) + (int64_t)g * M_max * K,
        reinterpret_cast<const int8_t*>(params.w_ptr) + (int64_t)g * N * K,
        reinterpret_cast<float*>(params.o_ptr) + (int64_t)g * M_max * N,
        params.sfa_ptr + (int64_t)g * M_max * kb,
        params.sfb_ptr + (int64_t)g * N * kb,
        len, N, K, kb,
        static_cast<int>(blockIdx.x), static_cast<int>(threadIdx.x), smem_);

#else
    if (blockIdx.x == 0 && threadIdx.x == 0)
        printf("sm80_int8_asym_moe_gemm_masked_impl: requires __CUDA_ARCH__ >= 800\n");
#endif
}

}  // namespace asym_gemm
