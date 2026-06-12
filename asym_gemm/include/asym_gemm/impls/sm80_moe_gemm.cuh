// asym_gemm/include/asym_gemm/impls/sm80_moe_gemm.cuh
#pragma once

#include <cstdint>
#include <type_traits>

#include <cutlass/numeric_types.h>      // cutlass::half_t, cutlass::bfloat16_t
#include <cutlass/array.h>              // cutlass::Array<T, N>
#include <cutlass/numeric_conversion.h> // cutlass::NumericArrayConverter (cvt.rn.bf16x2.f32)

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
// moe_predicated_copy: vectorised tile copy with optional M-row predication.
//
// Port of FLASH_NAMESPACE::copy (flash-attention/csrc/flash_attn/src/utils.h):
// dispatches one TiledCopy atom per logical (m, k) sub-tile, gated by a per-row
// coordinate predicate.  When the underlying atom is 128-bit (UniversalCopy<uint128_t>
// or SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>), each call emits a single vectorised
// load/store instead of element-by-element scalar accesses.
//
// Template params:
//   Is_even_MN   — if true, all rows are valid; skip coordinate check (fast path).
//   Is_even_K    — if true, all K columns are valid; skip predicate_K check.
//   Clear_OOB_MN — if true, clear destination rows that are out-of-bounds.
//   Clear_OOB_K  — if true, clear destination columns that are out-of-bounds.
//
// Parameters:
//   tiled_copy   — TiledCopy object (must wrap a 128-bit atom for best perf).
//   S            — source tensor, rank-3: (Atom, MMA_M, MMA_N).
//   D            — destination tensor, same shape as S.
//   identity_MN  — coordinate tensor; identity_MN(0, m, 0) gives the M-row index.
//   predicate_K  — bool tensor of length size<2>(S); only consulted when !Is_even_K.
//   max_MN       — exclusive upper bound on valid M rows (ignored when Is_even_MN).
// ──────────────────────────────────────────────────────────────────────────────
template <bool Is_even_MN = true, bool Is_even_K = true,
          bool Clear_OOB_MN = false, bool Clear_OOB_K = false,
          typename TiledCopy,
          typename Engine0, typename Layout0,
          typename Engine1, typename Layout1,
          typename Engine2, typename Layout2,
          typename Engine3, typename Layout3>
CUTE_DEVICE void moe_predicated_copy(
    TiledCopy tiled_copy,
    cute::Tensor<Engine0, Layout0> const& S,
    cute::Tensor<Engine1, Layout1>&       D,
    cute::Tensor<Engine2, Layout2> const& identity_MN,
    cute::Tensor<Engine3, Layout3> const& predicate_K,
    int max_MN = 0)
{
    static_assert(!(Clear_OOB_MN && !Clear_OOB_K),
                  "Clear_OOB_MN requires Clear_OOB_K");
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(S); ++m) {
        if (Is_even_MN || cute::get<0>(identity_MN(cute::_0{}, m, cute::_0{})) < max_MN) {
            CUTE_UNROLL
            for (int k = 0; k < cute::size<2>(S); ++k) {
                if (Is_even_K || predicate_K(k))
                    cute::copy(tiled_copy, S(cute::_, m, k), D(cute::_, m, k));
                else if (Clear_OOB_K)
                    cute::clear(D(cute::_, m, k));
            }
        } else if (Clear_OOB_MN) {
            cute::clear(D(cute::_, m, cute::_));
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// moe_convert_type: vectorised register-fragment conversion via cutlass
// NumericArrayConverter.
//
// Port of FLASH_NAMESPACE::convert_type (flash-attention/csrc/flash_attn/src/utils.h).
// For FP32 → BF16 the inner 2-elem CUTLASS specialisation emits one
// `cvt.rn.bf16x2.f32` per pair (SM80+); for FP32 → FP16 it emits `cvt.rn.f16x2.f32`.
// The N>2 specialisation unrolls to N/2 SIMD calls.  Same rounding mode
// (round-to-nearest-even) and bit-identical output as the equivalent scalar
// `Element(static_cast<float>(x))` per-element loop.
//
// Requirements:
//   - tensor must be a register-residency tensor (e.g., from partition_fragment_C
//     or make_tensor<T>(layout)) whose data() returns a contiguous register array.
//   - tensor's element count must be known at compile time.
//
// Usage:
//   Tensor rO = moe_convert_type<ElementOut>(tSrO);
// ──────────────────────────────────────────────────────────────────────────────
template <typename To_type, typename Engine, typename Layout>
CUTE_DEVICE auto moe_convert_type(cute::Tensor<Engine, Layout> const& tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(cute::size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    auto frag = convert_op(
        *reinterpret_cast<const cutlass::Array<From_type, numel>*>(tensor.data()));
    return cute::make_tensor(cute::make_rmem_ptr<To_type>(&frag), tensor.layout());
}

// ──────────────────────────────────────────────────────────────────────────────
// Main kernel
//
// Grid:  (ceil_div(N, BLOCK_N), list_size)
//   blockIdx.x = which N-tile this CTA handles
//   blockIdx.y = which expert entry (index into expert_list / index_list)
// Block: (NWARPS * 32, 1, 1)
//
// Algorithm (M-outer, K-inner — accumulates in FP32, writes once per M-tile):
//   expert derived from blockIdx.y (one CTA per expert — parallel across SMs)
//   for each M-tile m:
//     clear FP32 accumulator
//     for each K-tile k:
//       load sX [BLOCK_M, BLOCK_K] from x[e][m][k]  (cp.async or element copy)
//       load sW [BLOCK_N, BLOCK_K] from w[expert_id][n_tile][k]  (cp.async)
//       fence + wait + syncthreads
//       LDSM sX → registers, LDSM sW → registers
//       MMA: acc += sX_reg @ sW_reg^T
//       syncthreads
//     convert acc (fp32) → Element, stage in sO, write sO → gO (with M predicate)
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

    // ── Expert selection: one CTA per expert (blockIdx.y = expert entry index) ──
    {
        const int     expert_e  = static_cast<int>(blockIdx.y);
        const int32_t expert_id = params.expert_list[expert_e];
        const int64_t len_start = (expert_e == 0)
                                ? 0LL
                                : static_cast<int64_t>(params.index_list[expert_e - 1]);
        const int64_t len       = static_cast<int64_t>(params.index_list[expert_e]) - len_start;

        // Early exit for zero-token experts or gap segments before any barriers.
        if (expert_id < 0 || len == 0) return;

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
        // Output-side coordinate / K-predicate tensors (stable across M-tiles).
        Tensor cO       = make_identity_tensor(Shape<Int<BLOCK_M>, Int<BLOCK_N>>{});
        Tensor tOcO     = gmem_thr_copy_o.partition_S(cO);

        // Partition W once (same source for all m_tiles per k_tile)
        Tensor tWsW     = gmem_thr_copy_xw.partition_D(sW);
        Tensor tOsO_src = gmem_thr_copy_o.partition_S(sO);
        // X-side K-predicate vector (all true; K is BLOCK_K-aligned).  Used by the
        // partial-tile cp.async path; Is_even_K=true skips the actual check.
        Tensor tXpX     = make_tensor<bool>(make_shape(size<2>(gmem_thr_copy_xw.partition_D(sX))));
        cute::fill(tXpX, true);
        // N-column predicate vector (all true; N is BLOCK_N-aligned by API contract).
        Tensor tOpO     = make_tensor<bool>(make_shape(size<2>(tOsO_src)));
        cute::fill(tOpO, true);

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

                // Load X tile via cp.async; partial last tile uses per-row M predicate.
                Tensor tXgX_k = tXgX_m(_, _, _, k);
                if (m_actual == static_cast<int>(BLOCK_M)) {
                    // Full tile: vectorised cp.async, no predicate overhead.
                    cute::copy(gmem_tiled_copy_xw, tXgX_k, tXsX);
                } else {
                    // Partial last tile: cp.async for valid rows, cute::clear for OOB.
                    moe_predicated_copy</*Is_even_MN=*/false, /*Is_even_K=*/true,
                                        /*Clear_OOB_MN=*/true,  /*Clear_OOB_K=*/true>(
                        gmem_tiled_copy_xw, tXgX_k, tXsX, tXcX, tXpX, m_actual);
                }
                cp_async_fence();
                cp_async_wait<0>();

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

            // ── Convert FP32 accumulator → Element via cvt.rn.{bf16x2,f16x2}.f32 ──
            Tensor rO = moe_convert_type<Element>(tSrO);

            // ── Stage output in smem ──────────────────────────────────────────
            Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
            Tensor taccOsO = smem_thr_copy_O.partition_D(sO);
            cute::copy(smem_copy_O, taccOrO, taccOsO);
            __syncthreads();

            // ── Write sO → gO via vectorised, predicated 128-bit stores ──────
            Tensor gO_m = gO(_, _, m);
            Tensor tOgO = gmem_thr_copy_o.partition_D(gO_m);

            // sO (smem) → tOrO (registers) via 128-bit LDS.
            Tensor tOrO = make_tensor<Element>(shape(tOgO));
            cute::copy(gmem_tiled_copy_o, tOsO_src, tOrO);

            // tOrO (registers) → tOgO (global) via 128-bit STG.
            if (m_actual == static_cast<int>(BLOCK_M)) {
                moe_predicated_copy</*Is_even_MN=*/true,  /*Is_even_K=*/true>(
                    gmem_tiled_copy_o, tOrO, tOgO, tOcO, tOpO);
            } else {
                moe_predicated_copy</*Is_even_MN=*/false, /*Is_even_K=*/true,
                                    /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                    gmem_tiled_copy_o, tOrO, tOgO, tOcO, tOpO, m_actual);
            }

            __syncthreads();
        }  // M-tile loop
    }  // expert block

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
// Grid:  (ceil_div(N, BLOCK_N), list_size)
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
template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t NWARPS,
          bool USE_BLOCK_SCALES = false>
__global__ void sm89_moe_fp8_gemm_impl(SM89MoEFP8Params params) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890

    static_assert(BLOCK_K >= 32,       "BLOCK_K must be >= 32 (SM89 FP8 MMA K-atom)");
    static_assert(BLOCK_K % 32 == 0,   "BLOCK_K must be a multiple of 32");
    static_assert(BLOCK_M % (NWARPS * 16) == 0, "BLOCK_M must be divisible by NWARPS*16");
    static_assert(BLOCK_K < 128 || BLOCK_K % 128 == 0,
                  "BLOCK_K must be <128 or a multiple of 128 (scale k-group alignment)");

    // Block-scale mode: number of 128-element scale k-groups per K-tile, and
    // MMA K-atoms (32 elements each) per group. BLOCK_K < 128 → the whole tile
    // sits inside one k-group (128 % BLOCK_K == 0 holds for 64/32).
    constexpr int SCALE_K_GROUPS    = (BLOCK_K >= 128) ? static_cast<int>(BLOCK_K / 128) : 1;
    constexpr int K_ATOMS_PER_GROUP = static_cast<int>((SCALE_K_GROUPS > 1 ? 128u : BLOCK_K) / 32u);

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
    constexpr int SMEM_O_BYTES = BLOCK_M * BLOCK_N * sizeof(ElementOut);
    constexpr int SMEM_SA_OFFSET = SMEM_X_ELEMS + SMEM_W_ELEMS + SMEM_O_BYTES;

    extern __shared__ char smem_[];
    ElementIn*  smem_x  = reinterpret_cast<ElementIn*>(smem_);
    ElementIn*  smem_w  = smem_x + SMEM_X_ELEMS;
    ElementOut* smem_o  = reinterpret_cast<ElementOut*>(smem_w + SMEM_W_ELEMS);
    float*      smem_sa = reinterpret_cast<float*>(smem_ + SMEM_SA_OFFSET);
    float*      smem_rcs = smem_sa + BLOCK_M;    // block mode: 1/cs_first per row
    float*      smem_ratio = smem_rcs + BLOCK_M; // block mode: [SCALE_K_GROUPS-1, BLOCK_M] cs_g/cs_{g+1}

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

    // ── Hoist: O-side coordinate / predicate tensors (stable across M-tiles) ──
    Tensor tOcO_o   = gmem_thr_copy_o.partition_S(cO);
    Tensor tOsO_src = gmem_thr_copy_o.partition_S(sO);
    Tensor tOpO_o   = make_tensor<bool>(make_shape(size<2>(tOsO_src)));
    cute::fill(tOpO_o, true);

    // MMA-C-layout coordinates: same shape as tSrO, gives the in-tile (m, n)
    // coord of each accumulator element (block-scale seed rescaling needs m).
    Tensor tScO = thr_mma.partition_C(cO);

    // ── Hoist: X-side K-predicate vector for partial-tile cp.async ──────────
    // K is always BLOCK_K-aligned (validated at API boundary), so all entries are
    // true and Is_even_K=true skips the check entirely; tXpX is just a parameter
    // slot for moe_predicated_copy.
    Tensor tXpX = make_tensor<bool>(make_shape(size<2>(gmem_thr_copy_xw.partition_D(sX))));
    cute::fill(tXpX, true);

    // ── Helper: write BF16 rO register buffer → sO → gO ─────────────────────
    // When apply_scale=true, per-token scales are in smem_sa[0..BLOCK_M-1] and
    // per-expert scale is in sb_val.  Scales are applied in float precision.
    auto write_output = [&](auto& rO, auto& gO_m, int m_actual,
                            bool apply_scale, float sb_val) {
        // Stage 1: rO (BF16 registers) → sO (BF16 smem) via MMA-C layout copy.
        Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
        Tensor taccOsO = smem_thr_copy_O.partition_D(sO);
        cute::copy(smem_copy_O, taccOrO, taccOsO);
        __syncthreads();

        // Stage 2: sO (smem) → tOrO (registers) via 128-bit LDS.
        Tensor tOgO = gmem_thr_copy_o.partition_D(gO_m);
        Tensor tOrO = make_tensor<ElementOut>(shape(tOgO));
        cute::copy(gmem_tiled_copy_o, tOsO_src, tOrO);

        if (apply_scale) {
            CUTE_UNROLL
            for (int mi = 0; mi < size<1>(tOcO_o); mi++) {
                int m_coord = get<0>(tOcO_o(_0{}, mi, _0{}));
                float cs = smem_sa[m_coord] * sb_val;
                CUTE_UNROLL
                for (int ai = 0; ai < size<0>(tOcO_o); ai++)
                    CUTE_UNROLL
                    for (int ni = 0; ni < size<2>(tOcO_o); ni++)
                        tOrO(ai, mi, ni) = ElementOut(
                            static_cast<float>(tOrO(ai, mi, ni)) * cs);
            }
        }

        // Stage 3: tOrO (registers) → tOgO (global) via 128-bit STG.
        // Branch on m_actual: full tiles take the Is_even_MN=true fast path.
        if (m_actual == static_cast<int>(BLOCK_M)) {
            moe_predicated_copy</*Is_even_MN=*/true,  /*Is_even_K=*/true>(
                gmem_tiled_copy_o, tOrO, tOgO, tOcO_o, tOpO_o);
        } else {
            // Partial last tile: skip OOB rows (Clear_OOB_MN=false → no zeros).
            moe_predicated_copy</*Is_even_MN=*/false, /*Is_even_K=*/true,
                                /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                gmem_tiled_copy_o, tOrO, tOgO, tOcO_o, tOpO_o, m_actual);
        }
        __syncthreads();
    };

    // ── Expert selection: one CTA per expert (blockIdx.y = expert entry index) ──
    {
        const int     expert_e  = static_cast<int>(blockIdx.y);
        const int32_t expert_id = params.expert_list[expert_e];
        const int64_t len_start = (expert_e == 0)
                                ? 0LL
                                : static_cast<int64_t>(params.index_list[expert_e - 1]);
        const int64_t len       = static_cast<int64_t>(params.index_list[expert_e]) - len_start;
        const int     m_max     = static_cast<int>((len + BLOCK_M - 1) / BLOCK_M);
        const int     k_max     = static_cast<int>(K / BLOCK_K);

        // Early exit for zero-token experts
        if (len == 0) return;

        // Per-token/per-expert scale pointers (nullptr → use scalar fallback)
        const bool   use_tensor_scales = (params.scale_a_ptr != nullptr);
        const float* sa_base = use_tensor_scales
            ? params.scale_a_ptr + len_start
            : nullptr;
        const float  sb_val  = use_tensor_scales
            ? params.scale_b_ptr[expert_id]
            : 0.0f;
        const float  scalar_cs = params.scale_a * params.scale_b;

        // Block-scale mode: 1x128 activation x 128x128 weight scales, applied
        // per K-tile. BLOCK_N <= 128 and 128 % BLOCK_N == 0, so each CTA's
        // N-tile sits in exactly one n-group.
        constexpr bool use_block_scales = USE_BLOCK_SCALES;
        const int  ng_idx = (n_tile * static_cast<int>(BLOCK_N)) / 128;

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

        // ── Helper: load per-token scales into smem for one M-tile ───────────
        auto load_scales_for_m_tile = [&](int m) {
            if (!use_tensor_scales) return;
            const int m_base = m * static_cast<int>(BLOCK_M);
            if (tidx < static_cast<int>(BLOCK_M)) {
                int m_global = m_base + tidx;
                smem_sa[tidx] = (m_global < static_cast<int>(len))
                    ? sa_base[m_global] : 1.0f;
            }
        };

        // ── Helper: block mode — combined cs = sa[token, kg] * sb[e, ng, kg]
        // for every scale k-group covered by one K-tile. Stores:
        //   smem_sa[m]    = cs of the LAST group   (epilogue multiplies by it)
        //   smem_rcs[m]   = 1 / cs of the FIRST group (seed rescale)
        //   smem_ratio[g][m] = cs_g / cs_{g+1}     (inter-group accumulator fold)
        auto load_block_scales_for_tile = [&](int m, int k) {
            const int m_base = m * static_cast<int>(BLOCK_M);
            if (tidx < static_cast<int>(BLOCK_M)) {
                const int  m_global = m_base + tidx;
                const bool valid    = m_global < static_cast<int>(len);
                const int  kg_base  = (k * static_cast<int>(BLOCK_K)) / 128;
                float cs[SCALE_K_GROUPS];
                CUTE_UNROLL
                for (int g = 0; g < SCALE_K_GROUPS; ++g) {
                    const int kg = kg_base + g;
                    const float sb = params.scale_b_blk_ptr[
                        ((int64_t)expert_id * params.sb_ng + ng_idx) * params.sa_kg + kg];
                    cs[g] = valid
                        ? params.scale_a_blk_ptr[(len_start + m_global) * params.sa_kg + kg] * sb
                        : 1.0f;
                }
                smem_sa[tidx]  = cs[SCALE_K_GROUPS - 1];
                smem_rcs[tidx] = 1.0f / cs[0];
                CUTE_UNROLL
                for (int g = 0; g + 1 < SCALE_K_GROUPS; ++g)
                    smem_ratio[g * static_cast<int>(BLOCK_M) + tidx] = cs[g] / cs[g + 1];
            }
        };

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
                } else {
                    moe_predicated_copy</*Is_even_MN=*/false, /*Is_even_K=*/true,
                                        /*Clear_OOB_MN=*/true,  /*Clear_OOB_K=*/true>(
                        gmem_tiled_copy_xw, tXgX_mk, tXsX, tXcX, tXpX, m_actual);
                }
                if (use_block_scales) load_block_scales_for_tile(m, 0);
                else                  load_scales_for_m_tile(m);
                cp_async_fence();
                cp_async_wait<0>();
                __syncthreads();

                // LDSM: sX → tSrX,  sW → tOrW
                Tensor tSrX_view = smem_thr_copy_A.retile_D(tSrX);
                cute::copy(smem_copy_A, tSsX, tSrX_view);
                Tensor tOrW_view = smem_thr_copy_B.retile_D(tOrW);
                cute::copy(smem_copy_B, tOsW, tOrW_view);

                // Native FP8 MMA → FP32 accumulator.
                // Block mode: MMA per 128-element scale k-group, folding each
                // group's scale into the accumulator via cs_g/cs_{g+1}; the
                // epilogue applies cs_last once:
                //   acc = seed/cs_0; per g: +S_g then ×cs_g/cs_{g+1}; out = acc·cs_last
                if (use_block_scales) {
                    CUTE_UNROLL
                    for (int g = 0; g < SCALE_K_GROUPS; ++g) {
                        CUTE_UNROLL
                        for (int kk = 0; kk < K_ATOMS_PER_GROUP; ++kk) {
                            const int ka = g * K_ATOMS_PER_GROUP + kk;
                            cute::gemm(tiled_mma, tSrO,
                                       tSrX(_, _, ka), tOrW(_, _, ka), tSrO);
                        }
                        if (g + 1 < SCALE_K_GROUPS) {
                            CUTE_UNROLL
                            for (int i = 0; i < size(tSrO); i++)
                                tSrO(i) *= smem_ratio[g * static_cast<int>(BLOCK_M)
                                                      + get<0>(tScO(i))];
                        }
                    }
                } else {
                    cute::gemm(tiled_mma, tSrO, tSrX, tOrW, tSrO);
                }

                if (k_max == 1 && !use_tensor_scales && !use_block_scales) {
                    CUTE_UNROLL
                    for (int i = 0; i < size(tSrO); i++) tSrO(i) *= scalar_cs;
                }

                // FP32 → BF16 via cvt.rn.bf16x2.f32, then write sO → gO
                // Block mode: every K-tile's partials are scaled by its own cs
                // (cs already combined into smem_sa, so sb_val passes as 1).
                Tensor rO = moe_convert_type<ElementOut>(tSrO);
                Tensor gO_m = gO(_, _, m);
                write_output(rO, gO_m, m_actual,
                             /*apply_scale=*/use_block_scales ||
                                 (k_max == 1 && use_tensor_scales),
                             use_block_scales ? 1.0f : sb_val);
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

                // Block mode: this tile's cs/rcs must be in smem before the
                // seed rescale below (cross-thread read → needs a barrier).
                if (use_block_scales) {
                    load_block_scales_for_tile(m, k);
                    __syncthreads();
                }

                // Load X[m,k] → sX
                Tensor gX_m    = gX(_, _, m, _);
                Tensor gO_m    = gO(_, _, m);
                Tensor tXsX    = gmem_thr_copy_xw.partition_D(sX);
                Tensor tXgX_mk = gmem_thr_copy_xw.partition_S(gX_m(_, _, k));
                Tensor tXcX    = gmem_thr_copy_xw.partition_S(cX);

                // Seed the accumulator with the previous K-tiles' partial sum.
                // Block mode: partials hold the SCALED running sum; divide by
                // this tile's cs so the shared epilogue (x cs) restores it:
                //   O_k = cs_k * (O_{k-1} / cs_k + X_k W_k) = O_{k-1} + cs_k X_k W_k
                if (m_actual == static_cast<int>(BLOCK_M)) {
                    cute::copy(gmem_tiled_copy_xw, tXgX_mk, tXsX);
                    cp_async_fence();
                    cp_async_wait<0>();
                    Tensor tSgO = thr_mma.partition_C(gO_m);
                    if (use_block_scales) {
                        CUTE_UNROLL
                        for (int i = 0; i < size(tSrO); i++)
                            tSrO(i) = static_cast<float>(tSgO(i))
                                      * smem_rcs[get<0>(tScO(i))];
                    } else {
                        CUTE_UNROLL
                        for (int i = 0; i < size(tSrO); i++)
                            tSrO(i) = static_cast<float>(tSgO(i));
                    }
                } else {
                    moe_predicated_copy</*Is_even_MN=*/false, /*Is_even_K=*/true,
                                        /*Clear_OOB_MN=*/true,  /*Clear_OOB_K=*/true>(
                        gmem_tiled_copy_xw, tXgX_mk, tXsX, tXcX, tXpX, m_actual);
                    cp_async_fence();
                    cp_async_wait<0>();
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
                    __syncthreads();
                    Tensor rO_seed      = make_tensor<ElementOut>(shape(tSrO));
                    Tensor rO_seed_view = smem_thr_copy_O.retile_D(rO_seed);
                    cute::copy(smem_copy_O, tSsO, rO_seed_view);
                    if (use_block_scales) {
                        CUTE_UNROLL
                        for (int i = 0; i < size(tSrO); i++)
                            tSrO(i) = static_cast<float>(rO_seed(i))
                                      * smem_rcs[get<0>(tScO(i))];
                    } else {
                        CUTE_UNROLL
                        for (int i = 0; i < size(tSrO); i++)
                            tSrO(i) = static_cast<float>(rO_seed(i));
                    }
                }

                if (!use_block_scales && is_last_k) load_scales_for_m_tile(m);

                // LDSM: sX → tSrX,  sW → tOrW
                Tensor tSrX_view = smem_thr_copy_A.retile_D(tSrX);
                cute::copy(smem_copy_A, tSsX, tSrX_view);
                Tensor tOrW_view = smem_thr_copy_B.retile_D(tOrW);
                cute::copy(smem_copy_B, tOsW, tOrW_view);

                // Native FP8 MMA — accumulates into seeded FP32 tSrO.
                // Block mode: per-group MMA + inter-group rescale (see k=0 path).
                if (use_block_scales) {
                    CUTE_UNROLL
                    for (int g = 0; g < SCALE_K_GROUPS; ++g) {
                        CUTE_UNROLL
                        for (int kk = 0; kk < K_ATOMS_PER_GROUP; ++kk) {
                            const int ka = g * K_ATOMS_PER_GROUP + kk;
                            cute::gemm(tiled_mma, tSrO,
                                       tSrX(_, _, ka), tOrW(_, _, ka), tSrO);
                        }
                        if (g + 1 < SCALE_K_GROUPS) {
                            CUTE_UNROLL
                            for (int i = 0; i < size(tSrO); i++)
                                tSrO(i) *= smem_ratio[g * static_cast<int>(BLOCK_M)
                                                      + get<0>(tScO(i))];
                        }
                    }
                } else {
                    cute::gemm(tiled_mma, tSrO, tSrX, tOrW, tSrO);
                }

                if (is_last_k && !use_tensor_scales && !use_block_scales) {
                    CUTE_UNROLL
                    for (int i = 0; i < size(tSrO); i++) tSrO(i) *= scalar_cs;
                }

                // FP32 → BF16 via cvt.rn.bf16x2.f32, then write sO → gO
                Tensor rO = moe_convert_type<ElementOut>(tSrO);
                write_output(rO, gO_m, m_actual,
                             /*apply_scale=*/use_block_scales ||
                                 (is_last_k && use_tensor_scales),
                             use_block_scales ? 1.0f : sb_val);
            }  // m-loop (k>0)
        }  // k-loop
    }  // expert block

#else
    if (blockIdx.x == 0 && threadIdx.x == 0)
        printf("sm89_moe_fp8_gemm_impl: requires __CUDA_ARCH__ >= 890\n");
#endif
}

// ──────────────────────────────────────────────────────────────────────────────
// Masked FP8 MoE GEMM kernel — SM89 (RTX 4090) native FP8 MMA
//
// Grid:  (ceil_div(N, BLOCK_N), num_groups)     ← CUDA-graph safe (constant)
// Block: (NWARPS * 32, 1, 1)
//
// Input layout: a[G, M_max, K], b[G, N, K], d[G, M_max, N]
// Each CTA reads masked_m[blockIdx.y] to get the valid row count for its
// expert, then processes only those rows. Experts with 0 valid rows early-exit.
// ──────────────────────────────────────────────────────────────────────────────
template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t NWARPS,
          bool USE_BLOCK_SCALES = false>
__global__ void sm89_moe_fp8_gemm_masked_impl(SM89MoEFP8MaskedParams params) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890

    static_assert(BLOCK_K >= 32,       "BLOCK_K must be >= 32 (SM89 FP8 MMA K-atom)");
    static_assert(BLOCK_K % 32 == 0,   "BLOCK_K must be a multiple of 32");
    static_assert(BLOCK_M % (NWARPS * 16) == 0, "BLOCK_M must be divisible by NWARPS*16");
    static_assert(BLOCK_K < 128 || BLOCK_K % 128 == 0,
                  "BLOCK_K must be <128 or a multiple of 128 (scale k-group alignment)");

    // Block-scale mode: number of 128-element scale k-groups per K-tile, and
    // MMA K-atoms (32 elements each) per group. BLOCK_K < 128 → the whole tile
    // sits inside one k-group (128 % BLOCK_K == 0 holds for 64/32).
    constexpr int SCALE_K_GROUPS    = (BLOCK_K >= 128) ? static_cast<int>(BLOCK_K / 128) : 1;
    constexpr int K_ATOMS_PER_GROUP = static_cast<int>((SCALE_K_GROUPS > 1 ? 128u : BLOCK_K) / 32u);

    using ElementIn  = cutlass::float_e4m3_t;
    using ElementOut = cutlass::bfloat16_t;

    const ElementIn*  __restrict__ x_g = reinterpret_cast<const ElementIn*>(params.x_ptr);
    const ElementIn*  __restrict__ w_g = reinterpret_cast<const ElementIn*>(params.w_ptr);
    ElementOut*       __restrict__ o_g = reinterpret_cast<ElementOut*>(params.o_ptr);

    const int64_t M_max  = params.M_max;
    const int64_t N      = params.N;
    const int64_t K      = params.K;
    const int     tidx   = static_cast<int>(threadIdx.x);
    const int     n_tile = static_cast<int>(blockIdx.x);

    // ── Shared memory layout (identical to contiguous FP8 kernel) ────────────
    using SmemLayoutX = Layout<Shape<Int<BLOCK_M>, Int<BLOCK_K>>,
                               Stride<Int<BLOCK_K>, _1>>;
    using SmemLayoutW = Layout<Shape<Int<BLOCK_N>, Int<BLOCK_K>>,
                               Stride<Int<BLOCK_K>, _1>>;
    using SmemLayoutO = Layout<Shape<Int<BLOCK_M>, Int<BLOCK_N>>,
                               Stride<Int<BLOCK_N>, _1>>;

    constexpr int SMEM_X_ELEMS = BLOCK_M * BLOCK_K;
    constexpr int SMEM_W_ELEMS = BLOCK_N * BLOCK_K;
    constexpr int SMEM_O_BYTES = BLOCK_M * BLOCK_N * sizeof(ElementOut);
    constexpr int SMEM_SA_OFFSET = SMEM_X_ELEMS + SMEM_W_ELEMS + SMEM_O_BYTES;

    extern __shared__ char smem_[];
    ElementIn*  smem_x  = reinterpret_cast<ElementIn*>(smem_);
    ElementIn*  smem_w  = smem_x + SMEM_X_ELEMS;
    ElementOut* smem_o  = reinterpret_cast<ElementOut*>(smem_w + SMEM_W_ELEMS);
    float*      smem_sa = reinterpret_cast<float*>(smem_ + SMEM_SA_OFFSET);
    float*      smem_rcs = smem_sa + BLOCK_M;    // block mode: 1/cs_first per row
    float*      smem_ratio = smem_rcs + BLOCK_M; // block mode: [SCALE_K_GROUPS-1, BLOCK_M] cs_g/cs_{g+1}

    Tensor sX = make_tensor(make_smem_ptr(smem_x), SmemLayoutX{});
    Tensor sW = make_tensor(make_smem_ptr(smem_w), SmemLayoutW{});
    Tensor sO = make_tensor(make_smem_ptr(smem_o), SmemLayoutO{});

    // ── MMA setup ────────────────────────────────────────────────────────────
    using MMA_Op   = SM89_16x8x32_F32E4M3E4M3F32_TN;
    using TiledMma = TiledMMA<
        MMA_Atom<MMA_Op>,
        Layout<Shape<Int<NWARPS>, _1, _1>>,
        Tile<Int<16 * NWARPS>, _16, _32>>;

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);

    Tensor tSrO = thr_mma.partition_fragment_C(sO);
    Tensor tSrX = thr_mma.partition_fragment_A(sX);
    Tensor tOrW = thr_mma.partition_fragment_B(sW);

    // ── LDSM ─────────────────────────────────────────────────────────────────
    using SmemCopyAtomFP8 = Copy_Atom<SM75_U32x4_LDSM_N, ElementIn>;
    auto smem_copy_A      = make_tiled_copy_A(SmemCopyAtomFP8{}, tiled_mma);
    auto smem_copy_B      = make_tiled_copy_B(SmemCopyAtomFP8{}, tiled_mma);
    auto smem_thr_copy_A  = smem_copy_A.get_thread_slice(tidx);
    auto smem_thr_copy_B  = smem_copy_B.get_thread_slice(tidx);
    Tensor tSsX = smem_thr_copy_A.partition_S(sX);
    Tensor tOsW = smem_thr_copy_B.partition_S(sW);

    // ── C-side smem copy ─────────────────────────────────────────────────────
    using SmemCopyAtomO   = Copy_Atom<UniversalCopy<ElementOut>, ElementOut>;
    auto smem_copy_O      = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O  = smem_copy_O.get_thread_slice(tidx);
    Tensor tSsO = smem_thr_copy_O.partition_S(sO);

    // ── Global→smem copies ───────────────────────────────────────────────────
    using GmemCopyAtomFP8 = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, ElementIn>;
    using GmemTiledCopyXW = decltype(make_tiled_copy(
        GmemCopyAtomFP8{},
        Layout<Shape<_32, _4>, Stride<_4, _1>>{},
        Layout<Shape<_1, _16>>{}));
    GmemTiledCopyXW gmem_tiled_copy_xw;
    auto gmem_thr_copy_xw = gmem_tiled_copy_xw.get_thread_slice(tidx);

    using GmemCopyAtomBF16 = Copy_Atom<UniversalCopy<uint128_t>, ElementOut>;
    using GmemTiledCopyO   = decltype(make_tiled_copy(
        GmemCopyAtomBF16{},
        Layout<Shape<_32, _4>, Stride<_4, _1>>{},
        Layout<Shape<_1, _8>>{}));
    GmemTiledCopyO gmem_tiled_copy_o;
    auto gmem_thr_copy_o = gmem_tiled_copy_o.get_thread_slice(tidx);

    // Coordinate tensors for M-predication
    Tensor cX = make_identity_tensor(Shape<Int<BLOCK_M>, Int<BLOCK_K>>{});
    Tensor cO = make_identity_tensor(Shape<Int<BLOCK_M>, Int<BLOCK_N>>{});

    Tensor tOcO_o   = gmem_thr_copy_o.partition_S(cO);
    Tensor tOsO_src = gmem_thr_copy_o.partition_S(sO);
    Tensor tOpO_o   = make_tensor<bool>(make_shape(size<2>(tOsO_src)));
    cute::fill(tOpO_o, true);

    Tensor tXpX = make_tensor<bool>(make_shape(size<2>(gmem_thr_copy_xw.partition_D(sX))));
    cute::fill(tXpX, true);

    // MMA-C-layout coordinates: same shape as tSrO, gives the in-tile (m, n)
    // coord of each accumulator element (block-scale seed rescaling needs m).
    Tensor tScO = thr_mma.partition_C(cO);

    // ── Helper: write BF16 rO register buffer → sO → gO ─────────────────────
    // When apply_scale=true, per-token scales are in smem_sa[0..BLOCK_M-1] and
    // per-expert scale is in sb_val.  Scales are applied in float precision.
    auto write_output = [&](auto& rO, auto& gO_m, int m_actual,
                            bool apply_scale, float sb_val) {
        Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
        Tensor taccOsO = smem_thr_copy_O.partition_D(sO);
        cute::copy(smem_copy_O, taccOrO, taccOsO);
        __syncthreads();

        Tensor tOgO = gmem_thr_copy_o.partition_D(gO_m);
        Tensor tOrO = make_tensor<ElementOut>(shape(tOgO));
        cute::copy(gmem_tiled_copy_o, tOsO_src, tOrO);

        if (apply_scale) {
            CUTE_UNROLL
            for (int mi = 0; mi < size<1>(tOcO_o); mi++) {
                int m_coord = get<0>(tOcO_o(_0{}, mi, _0{}));
                float cs = smem_sa[m_coord] * sb_val;
                CUTE_UNROLL
                for (int ai = 0; ai < size<0>(tOcO_o); ai++)
                    CUTE_UNROLL
                    for (int ni = 0; ni < size<2>(tOcO_o); ni++)
                        tOrO(ai, mi, ni) = ElementOut(
                            static_cast<float>(tOrO(ai, mi, ni)) * cs);
            }
        }

        if (m_actual == static_cast<int>(BLOCK_M)) {
            moe_predicated_copy</*Is_even_MN=*/true,  /*Is_even_K=*/true>(
                gmem_tiled_copy_o, tOrO, tOgO, tOcO_o, tOpO_o);
        } else {
            moe_predicated_copy</*Is_even_MN=*/false, /*Is_even_K=*/true,
                                /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                gmem_tiled_copy_o, tOrO, tOgO, tOcO_o, tOpO_o, m_actual);
        }
        __syncthreads();
    };

    // ── Expert selection: blockIdx.y = group index, len from masked_m ────────
    {
        const int     expert_e = static_cast<int>(blockIdx.y);
        const int64_t len      = static_cast<int64_t>(params.masked_m[expert_e]);
        const int     m_max    = static_cast<int>((len + BLOCK_M - 1) / BLOCK_M);
        const int     k_max    = static_cast<int>(K / BLOCK_K);

        if (len == 0) return;

        // Padded layout: expert_e indexes into [G, M_max, K] and [G, M_max, N]
        const ElementIn* x_e = x_g + (int64_t)expert_e * M_max * K;
        const ElementIn* w_e = w_g + (int64_t)expert_e * N * K;
        ElementOut*      o_e = o_g + (int64_t)expert_e * M_max * N;

        // Per-token/per-expert scale pointers (nullptr → use scalar fallback)
        const bool   use_tensor_scales = (params.scale_a_ptr != nullptr);
        const float* sa_base = use_tensor_scales
            ? params.scale_a_ptr + (int64_t)expert_e * M_max
            : nullptr;
        const float  sb_val  = use_tensor_scales
            ? params.scale_b_ptr[expert_e]
            : 0.0f;
        const float  scalar_cs = params.scale_a * params.scale_b;

        // Block-scale mode: 1x128 activation x 128x128 weight scales, applied
        // per K-tile (see contiguous kernel for the seed-rescale scheme).
        constexpr bool use_block_scales = USE_BLOCK_SCALES;
        const int  ng_idx = (n_tile * static_cast<int>(BLOCK_N)) / 128;

        Tensor mX = make_tensor(make_gmem_ptr(x_e),
                                make_shape(len, K), make_stride(K, Int<1>{}));
        Tensor mW = make_tensor(make_gmem_ptr(w_e),
                                make_shape(N,   K), make_stride(K, Int<1>{}));
        Tensor mO = make_tensor(make_gmem_ptr(o_e),
                                make_shape(len, N), make_stride(N, Int<1>{}));

        Tensor gX = local_tile(mX, Shape<Int<BLOCK_M>, Int<BLOCK_K>>{}, make_coord(_, _));
        Tensor gW = local_tile(mW, Shape<Int<BLOCK_N>, Int<BLOCK_K>>{}, make_coord(n_tile, _));
        Tensor gO = local_tile(mO, Shape<Int<BLOCK_M>, Int<BLOCK_N>>{}, make_coord(_, n_tile));

        Tensor tWsW = gmem_thr_copy_xw.partition_D(sW);

        // ── Helper: load per-token scales into smem for one M-tile ───────────
        auto load_scales_for_m_tile = [&](int m) {
            if (!use_tensor_scales) return;
            const int m_base = m * static_cast<int>(BLOCK_M);
            if (tidx < static_cast<int>(BLOCK_M)) {
                int m_global = m_base + tidx;
                smem_sa[tidx] = (m_global < static_cast<int>(len))
                    ? sa_base[m_global] : 1.0f;
            }
        };

        // ── Helper: block mode — combined cs = sa[g, token, kg] * sb[g, ng, kg]
        // for every scale k-group covered by one K-tile (see contiguous kernel).
        auto load_block_scales_for_tile = [&](int m, int k) {
            const int m_base = m * static_cast<int>(BLOCK_M);
            if (tidx < static_cast<int>(BLOCK_M)) {
                const int  m_global = m_base + tidx;
                const bool valid    = m_global < static_cast<int>(len);
                const int  kg_base  = (k * static_cast<int>(BLOCK_K)) / 128;
                float cs[SCALE_K_GROUPS];
                CUTE_UNROLL
                for (int g = 0; g < SCALE_K_GROUPS; ++g) {
                    const int kg = kg_base + g;
                    const float sb = params.scale_b_blk_ptr[
                        ((int64_t)expert_e * params.sb_ng + ng_idx) * params.sa_kg + kg];
                    cs[g] = valid
                        ? params.scale_a_blk_ptr[
                              ((int64_t)expert_e * M_max + m_global) * params.sa_kg + kg] * sb
                        : 1.0f;
                }
                smem_sa[tidx]  = cs[SCALE_K_GROUPS - 1];
                smem_rcs[tidx] = 1.0f / cs[0];
                CUTE_UNROLL
                for (int g = 0; g + 1 < SCALE_K_GROUPS; ++g)
                    smem_ratio[g * static_cast<int>(BLOCK_M) + tidx] = cs[g] / cs[g + 1];
            }
        };

        // ── k = 0 path ───────────────────────────────────────────────────────
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

                Tensor gX_m    = gX(_, _, m, _);
                Tensor tXsX    = gmem_thr_copy_xw.partition_D(sX);
                Tensor tXgX_mk = gmem_thr_copy_xw.partition_S(gX_m(_, _, 0));
                Tensor tXcX    = gmem_thr_copy_xw.partition_S(cX);

                if (m_actual == static_cast<int>(BLOCK_M)) {
                    cute::copy(gmem_tiled_copy_xw, tXgX_mk, tXsX);
                } else {
                    moe_predicated_copy</*Is_even_MN=*/false, /*Is_even_K=*/true,
                                        /*Clear_OOB_MN=*/true,  /*Clear_OOB_K=*/true>(
                        gmem_tiled_copy_xw, tXgX_mk, tXsX, tXcX, tXpX, m_actual);
                }
                if (use_block_scales) load_block_scales_for_tile(m, 0);
                else                  load_scales_for_m_tile(m);
                cp_async_fence();
                cp_async_wait<0>();
                __syncthreads();

                Tensor tSrX_view = smem_thr_copy_A.retile_D(tSrX);
                cute::copy(smem_copy_A, tSsX, tSrX_view);
                Tensor tOrW_view = smem_thr_copy_B.retile_D(tOrW);
                cute::copy(smem_copy_B, tOsW, tOrW_view);

                // Block mode: per-group MMA + inter-group rescale (see contiguous).
                if (use_block_scales) {
                    CUTE_UNROLL
                    for (int g = 0; g < SCALE_K_GROUPS; ++g) {
                        CUTE_UNROLL
                        for (int kk = 0; kk < K_ATOMS_PER_GROUP; ++kk) {
                            const int ka = g * K_ATOMS_PER_GROUP + kk;
                            cute::gemm(tiled_mma, tSrO,
                                       tSrX(_, _, ka), tOrW(_, _, ka), tSrO);
                        }
                        if (g + 1 < SCALE_K_GROUPS) {
                            CUTE_UNROLL
                            for (int i = 0; i < size(tSrO); i++)
                                tSrO(i) *= smem_ratio[g * static_cast<int>(BLOCK_M)
                                                      + get<0>(tScO(i))];
                        }
                    }
                } else {
                    cute::gemm(tiled_mma, tSrO, tSrX, tOrW, tSrO);
                }

                if (k_max == 1 && !use_tensor_scales && !use_block_scales) {
                    CUTE_UNROLL
                    for (int i = 0; i < size(tSrO); i++) tSrO(i) *= scalar_cs;
                }

                Tensor rO = moe_convert_type<ElementOut>(tSrO);
                Tensor gO_m = gO(_, _, m);
                write_output(rO, gO_m, m_actual,
                             /*apply_scale=*/use_block_scales ||
                                 (k_max == 1 && use_tensor_scales),
                             use_block_scales ? 1.0f : sb_val);
            }
        }

        // ── k > 0 path ───────────────────────────────────────────────────────
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

                // Block mode: this tile's cs/rcs must be in smem before the
                // seed rescale below (cross-thread read → needs a barrier).
                if (use_block_scales) {
                    load_block_scales_for_tile(m, k);
                    __syncthreads();
                }

                Tensor gX_m    = gX(_, _, m, _);
                Tensor gO_m    = gO(_, _, m);
                Tensor tXsX    = gmem_thr_copy_xw.partition_D(sX);
                Tensor tXgX_mk = gmem_thr_copy_xw.partition_S(gX_m(_, _, k));
                Tensor tXcX    = gmem_thr_copy_xw.partition_S(cX);

                // Seed accumulator; block mode rescales by 1/cs (see contiguous).
                if (m_actual == static_cast<int>(BLOCK_M)) {
                    cute::copy(gmem_tiled_copy_xw, tXgX_mk, tXsX);
                    cp_async_fence();
                    cp_async_wait<0>();
                    Tensor tSgO = thr_mma.partition_C(gO_m);
                    if (use_block_scales) {
                        CUTE_UNROLL
                        for (int i = 0; i < size(tSrO); i++)
                            tSrO(i) = static_cast<float>(tSgO(i))
                                      * smem_rcs[get<0>(tScO(i))];
                    } else {
                        CUTE_UNROLL
                        for (int i = 0; i < size(tSrO); i++)
                            tSrO(i) = static_cast<float>(tSgO(i));
                    }
                } else {
                    moe_predicated_copy</*Is_even_MN=*/false, /*Is_even_K=*/true,
                                        /*Clear_OOB_MN=*/true,  /*Clear_OOB_K=*/true>(
                        gmem_tiled_copy_xw, tXgX_mk, tXsX, tXcX, tXpX, m_actual);
                    cp_async_fence();
                    cp_async_wait<0>();
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
                    __syncthreads();
                    Tensor rO_seed      = make_tensor<ElementOut>(shape(tSrO));
                    Tensor rO_seed_view = smem_thr_copy_O.retile_D(rO_seed);
                    cute::copy(smem_copy_O, tSsO, rO_seed_view);
                    if (use_block_scales) {
                        CUTE_UNROLL
                        for (int i = 0; i < size(tSrO); i++)
                            tSrO(i) = static_cast<float>(rO_seed(i))
                                      * smem_rcs[get<0>(tScO(i))];
                    } else {
                        CUTE_UNROLL
                        for (int i = 0; i < size(tSrO); i++)
                            tSrO(i) = static_cast<float>(rO_seed(i));
                    }
                }

                if (!use_block_scales && is_last_k) load_scales_for_m_tile(m);

                Tensor tSrX_view = smem_thr_copy_A.retile_D(tSrX);
                cute::copy(smem_copy_A, tSsX, tSrX_view);
                Tensor tOrW_view = smem_thr_copy_B.retile_D(tOrW);
                cute::copy(smem_copy_B, tOsW, tOrW_view);

                // Block mode: per-group MMA + inter-group rescale (see contiguous).
                if (use_block_scales) {
                    CUTE_UNROLL
                    for (int g = 0; g < SCALE_K_GROUPS; ++g) {
                        CUTE_UNROLL
                        for (int kk = 0; kk < K_ATOMS_PER_GROUP; ++kk) {
                            const int ka = g * K_ATOMS_PER_GROUP + kk;
                            cute::gemm(tiled_mma, tSrO,
                                       tSrX(_, _, ka), tOrW(_, _, ka), tSrO);
                        }
                        if (g + 1 < SCALE_K_GROUPS) {
                            CUTE_UNROLL
                            for (int i = 0; i < size(tSrO); i++)
                                tSrO(i) *= smem_ratio[g * static_cast<int>(BLOCK_M)
                                                      + get<0>(tScO(i))];
                        }
                    }
                } else {
                    cute::gemm(tiled_mma, tSrO, tSrX, tOrW, tSrO);
                }

                if (is_last_k && !use_tensor_scales && !use_block_scales) {
                    CUTE_UNROLL
                    for (int i = 0; i < size(tSrO); i++) tSrO(i) *= scalar_cs;
                }

                Tensor rO = moe_convert_type<ElementOut>(tSrO);
                write_output(rO, gO_m, m_actual,
                             /*apply_scale=*/use_block_scales ||
                                 (is_last_k && use_tensor_scales),
                             use_block_scales ? 1.0f : sb_val);
            }
        }
    }

#else
    if (blockIdx.x == 0 && threadIdx.x == 0)
        printf("sm89_moe_fp8_gemm_masked_impl: requires __CUDA_ARCH__ >= 890\n");
#endif
}

}  // namespace asym_gemm
