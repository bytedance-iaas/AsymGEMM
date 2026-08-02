// Copyright (c) 2026.

#include "../apis/exp_act_offload.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace asym_gemm::exp_act_offload {
namespace {

constexpr int kThreads = 256;

// v13: opt-out switch. The tiled, atomic-free LoRA-A weight-gradient kernel is
// the default; set ASYMM_LORA_A_GRAD_ATOMIC=1 to fall back to the legacy
// per-element atomicAdd kernel (kept for debugging / A-B comparison).
inline bool use_atomic_lora_a_grad() {
    const char* v = std::getenv("ASYMM_LORA_A_GRAD_ATOMIC");
    return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
}

__device__ __forceinline__ float to_float(at::BFloat16 v) {
    return static_cast<float>(v);
}

__device__ __forceinline__ at::BFloat16 to_bf16(float v) {
    return static_cast<at::BFloat16>(v);
}

int blocks_for(int64_t total) {
    if (total <= 0) return 0;
    return static_cast<int>((total + kThreads - 1) / kThreads);
}

void check_cuda_bf16_2d(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.dim() == 2, name, " must be 2D");
    TORCH_CHECK(t.scalar_type() == torch::kBFloat16, name, " must be BF16");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_cuda_bf16_3d(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.dim() == 3, name, " must be 3D");
    TORCH_CHECK(t.scalar_type() == torch::kBFloat16, name, " must be BF16");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_cpu_bf16_2d(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.device().is_cpu(), name, " must be CPU");
    TORCH_CHECK(t.dim() == 2, name, " must be 2D");
    TORCH_CHECK(t.scalar_type() == torch::kBFloat16, name, " must be BF16");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.is_pinned(), name, " must be pinned CPU memory");
}

void check_offsets_experts(const torch::Tensor& offsets, const torch::Tensor& experts, int64_t list_size) {
    TORCH_CHECK(offsets.is_cuda(), "offsets must be CUDA");
    TORCH_CHECK(experts.is_cuda(), "experts must be CUDA");
    TORCH_CHECK(offsets.dim() == 1 && experts.dim() == 1, "offsets/experts must be 1D");
    TORCH_CHECK(offsets.scalar_type() == torch::kInt32, "offsets must be int32");
    TORCH_CHECK(experts.scalar_type() == torch::kInt32, "experts must be int32");
    TORCH_CHECK(offsets.is_contiguous() && experts.is_contiguous(), "offsets/experts must be contiguous");
    TORCH_CHECK(list_size >= 1, "list_size must include the sentinel entry");
    TORCH_CHECK(experts.numel() >= list_size, "experts shorter than list_size");
    TORCH_CHECK(offsets.numel() >= 2 * (list_size - 1), "offsets must be pair offsets [2 * groups]");
}

struct GroupPlan {
    int64_t groups;
    int64_t max_rows;
};

GroupPlan validate_group_plan(
    const torch::Tensor& offsets,
    const torch::Tensor& experts,
    int64_t list_size,
    int64_t rows,
    int64_t num_experts) {
    check_offsets_experts(offsets, experts, list_size);
    const int64_t groups = list_size - 1;
    auto offsets_cpu = offsets.to(torch::kCPU);
    auto experts_cpu = experts.to(torch::kCPU);
    const auto offsets_acc = offsets_cpu.accessor<int32_t, 1>();
    const auto experts_acc = experts_cpu.accessor<int32_t, 1>();
    int64_t max_rows = 0;
    for (int64_t g = 0; g < groups; ++g) {
        const int64_t start = offsets_acc[2 * g];
        const int64_t end = offsets_acc[2 * g + 1];
        const int64_t expert = experts_acc[g];
        TORCH_CHECK(0 <= start && start <= end && end <= rows, "invalid grouped row offsets");
        TORCH_CHECK(expert >= 0 && expert < num_experts, "invalid expert id in grouped metadata");
        max_rows = std::max<int64_t>(max_rows, end - start);
    }
    return {.groups = groups, .max_rows = max_rows};
}

__global__ void lora_a_grad_kernel(
    const at::BFloat16* __restrict__ dS0,
    const at::BFloat16* __restrict__ dS1,
    const at::BFloat16* __restrict__ source_cpu,
    float* __restrict__ grad0_acc,
    float* __restrict__ grad1_acc,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ experts,
    int32_t groups,
    int32_t rows_total,
    int32_t rank,
    int32_t k_total,
    int64_t max_linear) {
    const int32_t group = static_cast<int32_t>(blockIdx.y);
    if (group >= groups) return;
    const int32_t expert = experts[group];
    const int32_t start = offsets[2 * group];
    const int32_t end = offsets[2 * group + 1];
    if (expert < 0 || start >= end) return;

    const int64_t rows = static_cast<int64_t>(end - start);
    const int64_t total = rows * static_cast<int64_t>(k_total);
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total && linear < max_linear;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int32_t k = static_cast<int32_t>(linear % k_total);
        const int32_t row = start + static_cast<int32_t>(linear / k_total);
        const float x = to_float(source_cpu[static_cast<int64_t>(row) * k_total + k]);
        const int64_t ds_base = static_cast<int64_t>(row) * rank;
        const int64_t grad_base = static_cast<int64_t>(expert) * rank * k_total + k;
        for (int32_t r = 0; r < rank; ++r) {
            atomicAdd(&grad0_acc[grad_base + static_cast<int64_t>(r) * k_total], to_float(dS0[ds_base + r]) * x);
            if (dS1 != nullptr && grad1_acc != nullptr) {
                atomicAdd(&grad1_acc[grad_base + static_cast<int64_t>(r) * k_total], to_float(dS1[ds_base + r]) * x);
            }
        }
    }
}

// v13: shared-memory-tiled, atomic-free LoRA-A weight-gradient kernel.
// v14: pair-salvage rework (goal 2, agent/impls/aymlora_kernels.md):
//   * PAIR is a compile-time template parameter — the single-output
//     instantiation carries no acc1/sS1 register or SMEM cost at all.
//   * RANK_EXACT specialization for the shipped rank (r == RANK_MAX): the
//     accumulator array has no dead guarded slots and the rank loops unroll
//     without runtime bounds checks.
//   * The next row-chunk's X (host-resident) and dS (HBM) tiles are
//     prefetched into REGISTERS while the current chunk computes, then
//     spilled to SMEM after the barrier. The NVLink-C2C load latency now
//     overlaps the FMA phase instead of serializing with it — this is what
//     lets the pair form add its second accumulator chain without adding
//     wall time (the "no mid-stream retirement" register-stationary
//     invariant is untouched: acc lives across the whole segment stream).
//
// Computes, per expert group g:  grad[g] = dS_g^T @ X_g   (shape [rank, K]),
// reducing over the group's rows *inside one CTA* so there are no global
// atomics. The output region [rank, K-tile] of each (group, K-tile) CTA is
// disjoint, so the FP32 register accumulators are written straight to BF16 ---
// no FP32 scratch buffer and no separate cast kernel are needed.
//
// Grid:  x = ceil(k_total / BN)  (output column tiles)
//        y = groups              (one expert segment each)
// Block: (BN, RY) threads. Thread (tx,ty) owns output column n0+tx and the
// rank rows r = ty, ty+RY, ... . X (CPU-resident) and dS (HBM) tiles are staged
// in SMEM and reused across the rank dimension.
// v16 (N0, stream-split): K2's reduction runs OVER the streamed axis, so at
// few segments the (k-tiles × groups) grid starves the GPU (measured 54 GB/s
// at one group vs the 211 ceiling). Parallelism must come from SPLITTING THE
// STREAM ITSELF: gridDim.z = S sub-streams per segment; each CTA keeps its
// [rank × k-tile] accumulator register-stationary for its WHOLE sub-stream
// and retires exactly once — hierarchical stream-end retirement (per
// sub-stream residency + one fp32 merge). S==1 (grid.z==1) takes the
// original write path unchanged. Upstream never needs this grid: its
// reduction axis is resident, only LoRA's training dataflow forces it.
template <int BN, int RY, int BROWS, int RANK_MAX, bool RANK_EXACT, bool PAIR>
__global__ void __launch_bounds__(BN * RY, 2) lora_a_grad_tiled_kernel(
    const at::BFloat16* __restrict__ dS0,
    const at::BFloat16* __restrict__ dS1,
    const at::BFloat16* __restrict__ source_cpu,
    at::BFloat16* __restrict__ grad0,
    at::BFloat16* __restrict__ grad1,
    float* __restrict__ part0,      // [S, groups, rank, k_total] when gridDim.z > 1
    float* __restrict__ part1,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ experts,
    int32_t groups,
    int32_t rank,
    int32_t k_total) {
    const int group = static_cast<int>(blockIdx.y);
    if (group >= groups) return;
    const int expert = experts[group];
    if (expert < 0) return;
    const int seg_start = offsets[2 * group];
    const int seg_end = offsets[2 * group + 1];
    const int seg_rows = seg_end - seg_start;
    // Sub-stream slice for this CTA (BROWS-aligned so chunk phases match the
    // single-split kernel exactly).
    const int splits = static_cast<int>(gridDim.z);
    const int split = static_cast<int>(blockIdx.z);
    const int chunks_total = (seg_rows + BROWS - 1) / BROWS;
    const int chunks_per_split = (chunks_total + splits - 1) / splits;
    const int start = seg_start + split * chunks_per_split * BROWS;
    const int end = min(seg_end, start + chunks_per_split * BROWS);
    const int rows = end - start;
    // rows<=0 CTAs still fall through: their zero accumulators must land in
    // the partial slice (the merge sums every split without pre-zeroing).
    const bool write_partial = splits > 1;

    const int tx = static_cast<int>(threadIdx.x);   // column within the tile
    const int ty = static_cast<int>(threadIdx.y);   // rank row-group
    const int n0 = static_cast<int>(blockIdx.x) * BN;
    const int col = n0 + tx;
    const bool col_valid = col < k_total;
    constexpr int NT = BN * RY;
    const int tid = ty * BN + tx;

    constexpr int ACC = (RANK_MAX + RY - 1) / RY;
    float acc0[ACC];
    float acc1[PAIR ? ACC : 1];
    #pragma unroll
    for (int i = 0; i < ACC; ++i) acc0[i] = 0.f;
    if constexpr (PAIR) {
        #pragma unroll
        for (int i = 0; i < ACC; ++i) acc1[i] = 0.f;
    }

    // Staged tiles live in SMEM as FP32 and TRANSPOSED (row-minor): the
    // bf16->f32 conversion happens once per element at staging time, and the
    // inner loop walks the row (reduction) axis with float4 loads — one
    // vector load per 4 rows instead of 4 scalar loads. This is what lets the
    // PAIR form's doubled read stream fit in the same wall time as one X
    // stream: the per-row instruction budget, not the link, was the pair
    // bottleneck. RPAD=4 keeps 16-byte float4 alignment while spreading the
    // lane-varying sX accesses across all 32 banks (stride 36 words: lane l
    // covers banks 4l..4l+3 mod 32 — conflict-free per 8-lane phase).
    constexpr int RPAD = 4;
    constexpr int RSTRIDE = BROWS + RPAD;
    __shared__ __align__(16) float sX[BN][RSTRIDE];
    __shared__ __align__(16) float sS0[RANK_MAX][RSTRIDE];
    __shared__ __align__(16) float sS1[PAIR ? RANK_MAX : 1][RSTRIDE];

    // Per-thread register slices of one staged chunk. X: 128-bit (8 bf16)
    // vectors over NVLink-C2C; dS: 32-bit (2 bf16) pieces from HBM. Sizes are
    // exact for the shipped shapes (BROWS*BN and BROWS*rank both divide NT
    // evenly); the generic path keeps per-element guards.
    constexpr int VEC = 8;
    constexpr int VPR = BN / VEC;                    // int4 chunks per tile row
    constexpr int XV = (BROWS * VPR + NT - 1) / NT;  // int4s per thread
    constexpr int SV_MAX = (BROWS * RANK_MAX / 2 + NT - 1) / NT;  // b32s per thread
    int4 xreg[XV];
    uint32_t s0reg[SV_MAX];
    uint32_t s1reg[PAIR ? SV_MAX : 1];
    const int sv = RANK_EXACT ? SV_MAX
                              : (BROWS * static_cast<int>(rank) / 2 + NT - 1) / NT;

    // Issue the global loads for row-chunk [r0, r0+crows) into registers.
    auto fetch_chunk = [&](int r0, int crows) {
        #pragma unroll
        for (int v = 0; v < XV; ++v) {
            const int e = tid + v * NT;
            const int rr = e / VPR;
            const int j = e - rr * VPR;
            const int c = n0 + j * VEC;
            if (rr < crows && c + VEC <= k_total) {
                xreg[v] = *reinterpret_cast<const int4*>(
                    &source_cpu[static_cast<int64_t>(start + r0 + rr) * k_total + c]);
            }
        }
        const int rank_run = RANK_EXACT ? RANK_MAX : static_cast<int>(rank);
        const int hpr = rank_run / 2;                // b32 pieces per row
        #pragma unroll
        for (int v = 0; v < SV_MAX; ++v) {
            if (!RANK_EXACT && v >= sv) break;
            const int e = tid + v * NT;
            const int rr = e / hpr;
            const int h = e - rr * hpr;
            if (rr < crows) {
                const int64_t base = static_cast<int64_t>(start + r0 + rr) * rank_run + 2 * h;
                s0reg[v] = *reinterpret_cast<const uint32_t*>(&dS0[base]);
                if constexpr (PAIR)
                    s1reg[v] = *reinterpret_cast<const uint32_t*>(&dS1[base]);
            }
        }
    };

    // Spill the registered chunk into SMEM, widening bf16 -> f32 on the way
    // (stalls until the loads land — by then the previous chunk's compute has
    // been overlapping them).
    auto cvt2 = [](uint32_t packed) {
        return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&packed));
    };
    auto store_chunk = [&](int crows) {
        #pragma unroll
        for (int v = 0; v < XV; ++v) {
            const int e = tid + v * NT;
            const int rr = e / VPR;
            const int j = e - rr * VPR;
            if (rr < crows && n0 + j * VEC + VEC <= k_total) {
                const uint32_t* p = reinterpret_cast<const uint32_t*>(&xreg[v]);
                #pragma unroll
                for (int h = 0; h < VEC / 2; ++h) {
                    const float2 f = cvt2(p[h]);
                    sX[j * VEC + 2 * h][rr] = f.x;
                    sX[j * VEC + 2 * h + 1][rr] = f.y;
                }
            }
        }
        const int rank_run = RANK_EXACT ? RANK_MAX : static_cast<int>(rank);
        const int hpr = rank_run / 2;
        #pragma unroll
        for (int v = 0; v < SV_MAX; ++v) {
            if (!RANK_EXACT && v >= sv) break;
            const int e = tid + v * NT;
            const int rr = e / hpr;
            const int h = e - rr * hpr;
            if (rr < crows) {
                const float2 f0 = cvt2(s0reg[v]);
                sS0[2 * h][rr] = f0.x;
                sS0[2 * h + 1][rr] = f0.y;
                if constexpr (PAIR) {
                    const float2 f1 = cvt2(s1reg[v]);
                    sS1[2 * h][rr] = f1.x;
                    sS1[2 * h + 1][rr] = f1.y;
                }
            }
        }
    };

    // Scalar-tail staging for exotic k_total not a multiple of 8 (kept off the
    // fast path; real shapes never take it).
    auto stage_x_tail = [&](int r0, int crows) {
        for (int e = tid; e < crows * VPR; e += NT) {
            const int rr = e / VPR;
            const int j = e - rr * VPR;
            const int c = n0 + j * VEC;
            if (c + VEC > k_total) {
                const int64_t xbase = static_cast<int64_t>(start + r0 + rr) * k_total + c;
                #pragma unroll
                for (int l = 0; l < VEC; ++l)
                    sX[j * VEC + l][rr] =
                        (c + l < k_total) ? static_cast<float>(source_cpu[xbase + l]) : 0.f;
            }
        }
    };
    const bool has_x_tail = (k_total % VEC) != 0 && (n0 + BN > k_total);

    const int crows0 = min(BROWS, rows);
    fetch_chunk(0, crows0);
    store_chunk(crows0);
    if (has_x_tail) stage_x_tail(0, crows0);
    __syncthreads();

    for (int r0 = 0; r0 < rows; r0 += BROWS) {
        const int crows = min(BROWS, rows - r0);
        const int next_r0 = r0 + BROWS;
        const int next_crows = min(BROWS, rows - next_r0);
        if (next_crows > 0)
            fetch_chunk(next_r0, next_crows);   // in flight during compute below
        if (col_valid) {
            // float4 over 4 rows at a time; per-slot loads are warp-broadcast
            // (all lanes read the same address). The 4 FMAs stay sequential so
            // the accumulation order — and thus the output bits — match the
            // scalar form exactly.
            const int crows4 = crows & ~3;
            for (int rr = 0; rr < crows4; rr += 4) {
                const float4 x4 = *reinterpret_cast<const float4*>(&sX[tx][rr]);
                #pragma unroll
                for (int i = 0; i < ACC; ++i) {
                    const int r = ty + i * RY;
                    if (RANK_EXACT || r < rank) {
                        const float4 s4 = *reinterpret_cast<const float4*>(&sS0[r][rr]);
                        acc0[i] += s4.x * x4.x;
                        acc0[i] += s4.y * x4.y;
                        acc0[i] += s4.z * x4.z;
                        acc0[i] += s4.w * x4.w;
                        if constexpr (PAIR) {
                            const float4 t4 = *reinterpret_cast<const float4*>(&sS1[r][rr]);
                            acc1[i] += t4.x * x4.x;
                            acc1[i] += t4.y * x4.y;
                            acc1[i] += t4.z * x4.z;
                            acc1[i] += t4.w * x4.w;
                        }
                    }
                }
            }
            for (int rr = crows4; rr < crows; ++rr) {
                const float xv = sX[tx][rr];
                #pragma unroll
                for (int i = 0; i < ACC; ++i) {
                    const int r = ty + i * RY;
                    if (RANK_EXACT || r < rank) {
                        acc0[i] += sS0[r][rr] * xv;
                        if constexpr (PAIR)
                            acc1[i] += sS1[r][rr] * xv;
                    }
                }
            }
        }
        __syncthreads();
        if (next_crows > 0) {
            store_chunk(next_crows);
            if (has_x_tail) stage_x_tail(next_r0, next_crows);
            __syncthreads();
        }
    }

    if (col_valid) {
        if (write_partial) {
            // Stream-end retirement of this SUB-stream: one fp32 write per
            // accumulator into the [S, groups, r, K] slice; the merge kernel
            // performs the single cross-substream reduction.
            const int64_t base = ((static_cast<int64_t>(split) * groups + group) * rank) * k_total;
            #pragma unroll
            for (int i = 0; i < ACC; ++i) {
                const int r = ty + i * RY;
                if (RANK_EXACT || r < rank) {
                    const int64_t out = base + static_cast<int64_t>(r) * k_total + col;
                    part0[out] = acc0[i];
                    if constexpr (PAIR)
                        part1[out] = acc1[i];
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < ACC; ++i) {
                const int r = ty + i * RY;
                if (RANK_EXACT || r < rank) {
                    const int64_t out =
                        (static_cast<int64_t>(expert) * rank + r) * k_total + col;
                    grad0[out] = static_cast<at::BFloat16>(acc0[i]);
                    if constexpr (PAIR)
                        grad1[out] = static_cast<at::BFloat16>(acc1[i]);
                }
            }
        }
    }
}

// =============================================================================
// N2 — dual-dataflow stream (2026-07-30). ONE pass over the host-resident X
// feeds TWO dataflows simultaneously:
//   S  = X·Aᵀ  — forward output, retire-per-chunk with cross-k-tile fp32
//                reduction (each CTA owns one 64-col k-slice of the shared
//                width, so it contributes a partial dot for every row);
//   dA = dSᵀ·X — gradient output, register-stationary per sub-stream with the
//                N0 hierarchical stream-end merge.
// The streamed operand therefore has two consumers with OPPOSITE reduction
// axes (S reduces over the resident width, dA over the stream itself) — a
// structure no inference kernel possesses (theirs has exactly one consumer).
// Bytes: halves X link reads at any site where fwd(S) and grad(dA) co-occur
// (GC-recompute attention windows; dS = g·B is computable before either).
// v1: exact rank 64, single dS; numerics: S is a sum of 32 ordered fp32
// k-slice partials via atomicAdd (fp32; run-to-run order nondeterministic at
// the ulp level), dA identical to the N0 path.
template <int BN, int RY, int BROWS, int RANK>
__global__ void __launch_bounds__(BN * RY, 2) lora_a_dual_stream_kernel(
    const at::BFloat16* __restrict__ dS0,
    const at::BFloat16* __restrict__ source_cpu,
    const at::BFloat16* __restrict__ lora_a,   // [E, RANK, k_total]
    float* __restrict__ s_out,                 // [rows_total, RANK] fp32, pre-zeroed
    float* __restrict__ part0,                 // [S, groups, RANK, k_total]
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ experts,
    int32_t groups,
    int32_t k_total) {
    const int group = static_cast<int>(blockIdx.y);
    if (group >= groups) return;
    const int expert = experts[group];
    if (expert < 0) return;
    const int seg_start = offsets[2 * group];
    const int seg_end = offsets[2 * group + 1];
    const int seg_rows = seg_end - seg_start;
    const int splits = static_cast<int>(gridDim.z);
    const int split = static_cast<int>(blockIdx.z);
    const int chunks_total = (seg_rows + BROWS - 1) / BROWS;
    const int chunks_per_split = (chunks_total + splits - 1) / splits;
    const int start = seg_start + split * chunks_per_split * BROWS;
    const int end = min(seg_end, start + chunks_per_split * BROWS);
    const int rows = end - start;

    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int n0 = static_cast<int>(blockIdx.x) * BN;
    const int col = n0 + tx;
    const bool col_valid = col < k_total;
    constexpr int NT = BN * RY;
    const int tid = ty * BN + tx;

    constexpr int ACC = RANK / RY;
    float acc0[ACC];
    #pragma unroll
    for (int i = 0; i < ACC; ++i) acc0[i] = 0.f;

    constexpr int RPAD = 4;
    constexpr int RSTRIDE = BROWS + RPAD;
    constexpr int CPAD = 4;
    constexpr int CSTRIDE = BN + CPAD;
    __shared__ __align__(16) float sX[BN][RSTRIDE];      // transposed — BOTH consumers read it
    __shared__ __align__(16) float sS0[RANK][RSTRIDE];
    __shared__ __align__(16) __nv_bfloat16 sA[RANK][BN];  // A slice (bf16: 8 coeffs/ld)

    constexpr int VEC = 8;
    constexpr int VPR = BN / VEC;
    constexpr int XV = (BROWS * VPR + NT - 1) / NT;
    constexpr int SV = (BROWS * RANK / 2 + NT - 1) / NT;
    int4 xreg[XV];
    uint32_t s0reg[SV];

    auto cvt2 = [](uint32_t packed) {
        return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&packed));
    };

    // One-time A-slice staging (HBM, tiny): sA[r][c] f32.
    for (int e = tid; e < RANK * VPR; e += NT) {
        const int r = e / VPR;
        const int j = e - r * VPR;
        const int c = n0 + j * VEC;
        if (c + VEC <= k_total) {
            *reinterpret_cast<int4*>(&sA[r][j * VEC]) = *reinterpret_cast<const int4*>(
                &lora_a[(static_cast<int64_t>(expert) * RANK + r) * k_total + c]);
        } else {
            for (int l = 0; l < VEC; ++l)
                sA[r][j * VEC + l] = (c + l < k_total)
                    ? *reinterpret_cast<const __nv_bfloat16*>(
                          &lora_a[(static_cast<int64_t>(expert) * RANK + r) * k_total + c + l])
                    : __float2bfloat16(0.f);
        }
    }

    auto fetch_chunk = [&](int r0, int crows) {
        #pragma unroll
        for (int v = 0; v < XV; ++v) {
            const int e = tid + v * NT;
            const int rr = e / VPR;
            const int j = e - rr * VPR;
            const int c = n0 + j * VEC;
            if (rr < crows && c + VEC <= k_total) {
                xreg[v] = *reinterpret_cast<const int4*>(
                    &source_cpu[static_cast<int64_t>(start + r0 + rr) * k_total + c]);
            }
        }
        constexpr int hpr = RANK / 2;
        #pragma unroll
        for (int v = 0; v < SV; ++v) {
            const int e = tid + v * NT;
            const int rr = e / hpr;
            const int h = e - rr * hpr;
            if (rr < crows) {
                const int64_t base = static_cast<int64_t>(start + r0 + rr) * RANK + 2 * h;
                s0reg[v] = *reinterpret_cast<const uint32_t*>(&dS0[base]);
            }
        }
    };
    auto store_chunk = [&](int crows) {
        #pragma unroll
        for (int v = 0; v < XV; ++v) {
            const int e = tid + v * NT;
            const int rr = e / VPR;
            const int j = e - rr * VPR;
            if (rr < crows && n0 + j * VEC + VEC <= k_total) {
                const uint32_t* p = reinterpret_cast<const uint32_t*>(&xreg[v]);
                #pragma unroll
                for (int h = 0; h < VEC / 2; ++h) {
                    const float2 f = cvt2(p[h]);
                    sX[j * VEC + 2 * h][rr] = f.x;
                    sX[j * VEC + 2 * h + 1][rr] = f.y;
                }
            }
        }
        constexpr int hpr = RANK / 2;
        #pragma unroll
        for (int v = 0; v < SV; ++v) {
            const int e = tid + v * NT;
            const int rr = e / hpr;
            const int h = e - rr * hpr;
            if (rr < crows) {
                const float2 f0 = cvt2(s0reg[v]);
                sS0[2 * h][rr] = f0.x;
                sS0[2 * h + 1][rr] = f0.y;
            }
        }
    };
    auto stage_x_tail = [&](int r0, int crows) {
        for (int e = tid; e < crows * VPR; e += NT) {
            const int rr = e / VPR;
            const int j = e - rr * VPR;
            const int c = n0 + j * VEC;
            if (c + VEC > k_total) {
                const int64_t xbase = static_cast<int64_t>(start + r0 + rr) * k_total + c;
                #pragma unroll
                for (int l = 0; l < VEC; ++l) {
                    sX[j * VEC + l][rr] =
                        (c + l < k_total) ? static_cast<float>(source_cpu[xbase + l]) : 0.f;
                }
            }
        }
    };
    const bool has_x_tail = (k_total % VEC) != 0 && (n0 + BN > k_total);

    const int crows0 = min(BROWS, rows);
    fetch_chunk(0, crows0);
    store_chunk(crows0);
    if (has_x_tail) stage_x_tail(0, crows0);
    __syncthreads();

    for (int r0 = 0; r0 < rows; r0 += BROWS) {
        const int crows = min(BROWS, rows - r0);
        const int next_r0 = r0 + BROWS;
        const int next_crows = min(BROWS, rows - next_r0);
        if (next_crows > 0)
            fetch_chunk(next_r0, next_crows);
        // dA consumer (stream-end stationary) — identical loop to the N0 kernel.
        if (col_valid) {
            const int crows4 = crows & ~3;
            for (int rr = 0; rr < crows4; rr += 4) {
                const float4 x4 = *reinterpret_cast<const float4*>(&sX[tx][rr]);
                #pragma unroll
                for (int i = 0; i < ACC; ++i) {
                    const int r = ty + i * RY;
                    const float4 s4 = *reinterpret_cast<const float4*>(&sS0[r][rr]);
                    acc0[i] += s4.x * x4.x;
                    acc0[i] += s4.y * x4.y;
                    acc0[i] += s4.z * x4.z;
                    acc0[i] += s4.w * x4.w;
                }
            }
            for (int rr = crows4; rr < crows; ++rr) {
                const float xv = sX[tx][rr];
                #pragma unroll
                for (int i = 0; i < ACC; ++i)
                    acc0[i] += sS0[ty + i * RY][rr] * xv;
            }
        }
        // S consumer (retire-per-chunk), fused into the SAME transposed
        // layout as dA: thread = (r, 4-row block); the sA[r][c] coefficient is
        // a warp broadcast (whole warp shares r) and the sX[c][rr..rr+3]
        // float4 uses the identical conflict-free pattern as the dA loop —
        // the second dataflow rides the one staged copy of the stream.
        for (int it = tid; it < RANK * (BROWS / 4); it += NT) {
            const int r = it / (BROWS / 4);
            const int rb = (it - r * (BROWS / 4)) * 4;
            const int cols_here = min(BN, k_total - n0);
            if (rb < crows) {
                float4 acc4 = make_float4(0.f, 0.f, 0.f, 0.f);
                const int ch8 = cols_here & ~7;
                for (int c = 0; c < ch8; c += 8) {
                    // One broadcast int4 fetches 8 bf16 coefficients; the x4
                    // loads keep the dA loop's conflict-free lane pattern.
                    const int4 a8 = *reinterpret_cast<const int4*>(&sA[r][c]);
                    const uint32_t* ap = reinterpret_cast<const uint32_t*>(&a8);
                    #pragma unroll
                    for (int h = 0; h < 4; ++h) {
                        const float2 f = cvt2(ap[h]);
                        const float4 xa = *reinterpret_cast<const float4*>(&sX[c + 2 * h][rb]);
                        acc4.x += f.x * xa.x;
                        acc4.y += f.x * xa.y;
                        acc4.z += f.x * xa.z;
                        acc4.w += f.x * xa.w;
                        const float4 xb = *reinterpret_cast<const float4*>(&sX[c + 2 * h + 1][rb]);
                        acc4.x += f.y * xb.x;
                        acc4.y += f.y * xb.y;
                        acc4.z += f.y * xb.z;
                        acc4.w += f.y * xb.w;
                    }
                }
                for (int c = ch8; c < cols_here; ++c) {
                    const float a = __bfloat162float(sA[r][c]);
                    const float4 x4 = *reinterpret_cast<const float4*>(&sX[c][rb]);
                    acc4.x += a * x4.x;
                    acc4.y += a * x4.y;
                    acc4.z += a * x4.z;
                    acc4.w += a * x4.w;
                }
                const int64_t sb = static_cast<int64_t>(start + r0 + rb) * RANK + r;
                const float av[4] = {acc4.x, acc4.y, acc4.z, acc4.w};
                #pragma unroll
                for (int l = 0; l < 4; ++l)
                    if (rb + l < crows)
                        atomicAdd(&s_out[sb + static_cast<int64_t>(l) * RANK], av[l]);
            }
        }
        __syncthreads();
        if (next_crows > 0) {
            store_chunk(next_crows);
            if (has_x_tail) stage_x_tail(next_r0, next_crows);
            __syncthreads();
        }
    }

    // dA stream-end retirement (always via partials; the N0 merge finishes).
    if (col_valid) {
        const int64_t base = ((static_cast<int64_t>(split) * groups + group) * RANK) * k_total;
        #pragma unroll
        for (int i = 0; i < ACC; ++i) {
            const int r = ty + i * RY;
            part0[base + static_cast<int64_t>(r) * k_total + col] = acc0[i];
        }
    }
}

// N0 merge: sum the S per-substream fp32 partials for one (group, r, k)
// position and retire once to bf16 grad (keyed by the group's expert id).
__global__ void lora_a_grad_split_merge_kernel(
    const float* __restrict__ part0,
    const float* __restrict__ part1,
    at::BFloat16* __restrict__ grad0,
    at::BFloat16* __restrict__ grad1,
    const int32_t* __restrict__ experts,
    int32_t splits,
    int32_t groups,
    int32_t rank,
    int32_t k_total) {
    const int64_t per_group = static_cast<int64_t>(rank) * k_total;
    const int64_t total = static_cast<int64_t>(groups) * per_group;
    const bool pair = (part1 != nullptr && grad1 != nullptr);
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int group = static_cast<int>(linear / per_group);
        const int64_t rk = linear - static_cast<int64_t>(group) * per_group;
        const int expert = experts[group];
        if (expert < 0) continue;
        float s0 = 0.f, s1 = 0.f;
        const int64_t stride = static_cast<int64_t>(groups) * per_group;
        const int64_t base = static_cast<int64_t>(group) * per_group + rk;
        for (int s = 0; s < splits; ++s) {
            s0 += part0[base + s * stride];
            if (pair) s1 += part1[base + s * stride];
        }
        const int64_t out = static_cast<int64_t>(expert) * per_group + rk;
        grad0[out] = static_cast<at::BFloat16>(s0);
        if (pair) grad1[out] = static_cast<at::BFloat16>(s1);
    }
}

template <int BN, int RY, int BROWS, int RANK_MAX>
void launch_lora_a_grad_tiled(
    const at::BFloat16* dS0,
    const at::BFloat16* dS1,
    const at::BFloat16* source_cpu,
    at::BFloat16* grad0,
    at::BFloat16* grad1,
    const int32_t* offsets,
    const int32_t* experts,
    int groups,
    int rank,
    int k_total,
    cudaStream_t stream) {
    TORCH_CHECK(rank <= RANK_MAX, "lora_a_grad_tiled: rank ", rank, " exceeds RANK_MAX ", RANK_MAX);
    TORCH_CHECK(rank % 2 == 0, "lora_a_grad_tiled: rank must be even, got ", rank);
    dim3 block(BN, RY);
    const int k_tiles = (k_total + BN - 1) / BN;
    // N0 adaptive stream-split: fill ~2 CTAs/SM (the kernel's occupancy
    // limit). The 128-expert grouped cell yields S=1 — byte-identical to the
    // pre-split kernel. ASYMM_LORA_A_GRAD_SPLIT=1 pins legacy; =N pins S.
    int splits = 1;
    {
        const char* v = std::getenv("ASYMM_LORA_A_GRAD_SPLIT");
        if (v != nullptr && v[0] != '\0') {
            splits = std::max(1, std::atoi(v));
        } else {
            const auto* props = at::cuda::getCurrentDeviceProperties();
            const int target = 2 * props->multiProcessorCount;
            splits = std::max(1, target / std::max(1, k_tiles * groups));
        }
    }
    dim3 grid(static_cast<unsigned int>(k_tiles), static_cast<unsigned int>(groups),
              static_cast<unsigned int>(splits));
    const bool pair = (dS1 != nullptr && grad1 != nullptr);
    float* part0 = nullptr;
    float* part1 = nullptr;
    torch::Tensor ws0, ws1;
    if (splits > 1) {
        const auto opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
        ws0 = torch::empty({static_cast<int64_t>(splits) * groups * rank * k_total}, opts);
        part0 = ws0.data_ptr<float>();
        if (pair) {
            ws1 = torch::empty_like(ws0);
            part1 = ws1.data_ptr<float>();
        }
    }
    constexpr int RANK_FAST = 64;  // the shipped LoRA rank — exact specialization
    if (rank == RANK_FAST) {
        if (pair) {
            lora_a_grad_tiled_kernel<BN, RY, BROWS, RANK_FAST, true, true><<<grid, block, 0, stream>>>(
                dS0, dS1, source_cpu, grad0, grad1, part0, part1, offsets, experts, groups, rank, k_total);
        } else {
            lora_a_grad_tiled_kernel<BN, RY, BROWS, RANK_FAST, true, false><<<grid, block, 0, stream>>>(
                dS0, nullptr, source_cpu, grad0, nullptr, part0, nullptr, offsets, experts, groups, rank, k_total);
        }
    } else if (pair) {
        lora_a_grad_tiled_kernel<BN, RY, BROWS, RANK_MAX, false, true><<<grid, block, 0, stream>>>(
            dS0, dS1, source_cpu, grad0, grad1, part0, part1, offsets, experts, groups, rank, k_total);
    } else {
        lora_a_grad_tiled_kernel<BN, RY, BROWS, RANK_MAX, false, false><<<grid, block, 0, stream>>>(
            dS0, nullptr, source_cpu, grad0, nullptr, part0, nullptr, offsets, experts, groups, rank, k_total);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (splits > 1) {
        const int64_t total = static_cast<int64_t>(groups) * rank * k_total;
        lora_a_grad_split_merge_kernel<<<blocks_for(total), kThreads, 0, stream>>>(
            part0, part1, grad0, grad1, experts,
            static_cast<int32_t>(splits), static_cast<int32_t>(groups),
            static_cast<int32_t>(rank), static_cast<int32_t>(k_total));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

__global__ void lora_b_backward_kernel(
    const at::BFloat16* __restrict__ grad_out_cpu,
    const at::BFloat16* __restrict__ low_rank,
    const at::BFloat16* __restrict__ lora_b,
    float* __restrict__ dS_acc,
    float* __restrict__ grad_b_acc,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ experts,
    int32_t groups,
    int32_t rows_total,
    int32_t out_dim,
    int32_t rank,
    float scale,
    int64_t max_linear) {
    const int32_t group = static_cast<int32_t>(blockIdx.y);
    if (group >= groups) return;
    const int32_t expert = experts[group];
    const int32_t start = offsets[2 * group];
    const int32_t end = offsets[2 * group + 1];
    if (expert < 0 || start >= end) return;

    const int64_t rows = static_cast<int64_t>(end - start);
    const int64_t total = rows * static_cast<int64_t>(out_dim);
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total && linear < max_linear;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int32_t i = static_cast<int32_t>(linear % out_dim);
        const int32_t row = start + static_cast<int32_t>(linear / out_dim);
        const float g = scale * to_float(grad_out_cpu[static_cast<int64_t>(row) * out_dim + i]);
        const int64_t lr_base = static_cast<int64_t>(row) * rank;
        const int64_t b_base = (static_cast<int64_t>(expert) * out_dim + i) * rank;
        for (int32_t r = 0; r < rank; ++r) {
            const float s = to_float(low_rank[lr_base + r]);
            const float b = to_float(lora_b[b_base + r]);
            atomicAdd(&dS_acc[lr_base + r], g * b);
            atomicAdd(&grad_b_acc[b_base + r], g * s);
        }
    }
}

__global__ void cast_fp32_to_bf16_kernel(
    const float* __restrict__ src,
    at::BFloat16* __restrict__ dst,
    int64_t total) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= total) return;
    dst[linear] = to_bf16(src[linear]);
}

void cast_fp32_to_bf16(const torch::Tensor& src, const torch::Tensor& dst, cudaStream_t stream) {
    const int64_t total = src.numel();
    if (total == 0) return;
    cast_fp32_to_bf16_kernel<<<blocks_for(total), kThreads, 0, stream>>>(
        src.data_ptr<float>(),
        dst.data_ptr<at::BFloat16>(),
        total);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void require_sm100() {
    const auto props = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(props != nullptr && props->major == 10, "expert activation offload kernels require SM100");
}

}  // namespace

void sm100_grouped_lora_a_grad_bf16_cpu_right(
    const torch::Tensor& grad_low_rank,
    const torch::Tensor& source_cpu,
    const torch::Tensor& grad_a,
    const torch::Tensor& offsets,
    const torch::Tensor& experts,
    int64_t list_size) {
    check_cuda_bf16_2d(grad_low_rank, "grad_low_rank");
    check_cpu_bf16_2d(source_cpu, "source_cpu");
    check_cuda_bf16_3d(grad_a, "grad_a");
    TORCH_CHECK(grad_low_rank.size(0) == source_cpu.size(0), "grad/source row mismatch");
    TORCH_CHECK(grad_a.size(1) == grad_low_rank.size(1), "grad_a rank mismatch");
    TORCH_CHECK(grad_a.size(2) == source_cpu.size(1), "grad_a K mismatch");
    TORCH_CHECK(grad_low_rank.device() == grad_a.device(), "grad_low_rank and grad_a must share device");
    TORCH_CHECK(offsets.device() == grad_low_rank.device() && experts.device() == grad_low_rank.device(),
                "metadata must be on grad_low_rank CUDA device");

    const c10::cuda::CUDAGuard device_guard(grad_low_rank.device());
    require_sm100();
    const auto plan = validate_group_plan(offsets, experts, list_size, grad_low_rank.size(0), grad_a.size(0));
    if (plan.groups <= 0 || plan.max_rows <= 0) {
        grad_a.zero_();
        return;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (use_atomic_lora_a_grad()) {
        auto grad_acc = torch::zeros(grad_a.sizes(), grad_a.options().dtype(torch::kFloat32));
        const int64_t max_linear = plan.max_rows * source_cpu.size(1);
        dim3 grid(blocks_for(max_linear), static_cast<unsigned int>(plan.groups));
        lora_a_grad_kernel<<<grid, kThreads, 0, stream>>>(
            grad_low_rank.data_ptr<at::BFloat16>(),
            nullptr,
            source_cpu.data_ptr<at::BFloat16>(),
            grad_acc.data_ptr<float>(),
            nullptr,
            offsets.data_ptr<int32_t>(),
            experts.data_ptr<int32_t>(),
            static_cast<int32_t>(plan.groups),
            static_cast<int32_t>(grad_low_rank.size(0)),
            static_cast<int32_t>(grad_low_rank.size(1)),
            static_cast<int32_t>(source_cpu.size(1)),
            max_linear);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        cast_fp32_to_bf16(grad_acc, grad_a, stream);
        return;
    }
    // Tiled, atomic-free path (default). Experts that never appear as a group
    // are zeroed here; active experts are fully overwritten by the kernel.
    grad_a.zero_();
    launch_lora_a_grad_tiled<64, 8, 32, 128>(
        grad_low_rank.data_ptr<at::BFloat16>(),
        nullptr,
        source_cpu.data_ptr<at::BFloat16>(),
        grad_a.data_ptr<at::BFloat16>(),
        nullptr,
        offsets.data_ptr<int32_t>(),
        experts.data_ptr<int32_t>(),
        static_cast<int32_t>(plan.groups),
        static_cast<int32_t>(grad_low_rank.size(1)),
        static_cast<int32_t>(source_cpu.size(1)),
        stream);
}

// N2 public entry: one X pass -> S (fp32, caller-zeroed) + dA (bf16).
void sm100_grouped_lora_a_dual_bf16_cpu_right(
    const torch::Tensor& grad_low_rank,
    const torch::Tensor& source_cpu,
    const torch::Tensor& lora_a,
    const torch::Tensor& s_out,
    const torch::Tensor& grad_a,
    const torch::Tensor& offsets,
    const torch::Tensor& experts,
    int64_t list_size) {
    check_cuda_bf16_2d(grad_low_rank, "grad_low_rank");
    check_cpu_bf16_2d(source_cpu, "source_cpu");
    check_cuda_bf16_3d(lora_a, "lora_a");
    check_cuda_bf16_3d(grad_a, "grad_a");
    TORCH_CHECK(s_out.is_cuda() && s_out.scalar_type() == torch::kFloat32 && s_out.is_contiguous(),
                "s_out must be contiguous CUDA fp32");
    TORCH_CHECK(grad_low_rank.size(1) == 64 && lora_a.size(1) == 64, "dual kernel v1 requires rank 64");
    TORCH_CHECK(grad_low_rank.size(0) == source_cpu.size(0), "dS/source row mismatch");
    TORCH_CHECK(s_out.size(0) == source_cpu.size(0) && s_out.size(1) == 64, "s_out shape mismatch");
    TORCH_CHECK(lora_a.size(2) == source_cpu.size(1) && grad_a.size(2) == source_cpu.size(1), "K mismatch");
    const c10::cuda::CUDAGuard device_guard(grad_low_rank.device());
    require_sm100();
    const auto plan = validate_group_plan(offsets, experts, list_size, grad_low_rank.size(0), grad_a.size(0));
    if (plan.groups <= 0 || plan.max_rows <= 0) {
        grad_a.zero_();
        return;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr int BN = 64, RY = 8, BROWS = 32, RANK = 64;
    const int k_total = static_cast<int>(source_cpu.size(1));
    const int groups = static_cast<int>(plan.groups);
    const int k_tiles = (k_total + BN - 1) / BN;
    int splits = 1;
    {
        const char* v = std::getenv("ASYMM_LORA_A_GRAD_SPLIT");
        if (v != nullptr && v[0] != '\0') {
            splits = std::max(1, std::atoi(v));
        } else {
            const auto* props = at::cuda::getCurrentDeviceProperties();
            splits = std::max(1, (2 * props->multiProcessorCount) / std::max(1, k_tiles * groups));
        }
    }
    const auto opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    auto ws0 = torch::empty({static_cast<int64_t>(splits) * groups * RANK * k_total}, opts);
    grad_a.zero_();
    dim3 block(BN, RY);
    dim3 grid(k_tiles, groups, splits);
    lora_a_dual_stream_kernel<BN, RY, BROWS, RANK><<<grid, block, 0, stream>>>(
        grad_low_rank.data_ptr<at::BFloat16>(),
        source_cpu.data_ptr<at::BFloat16>(),
        lora_a.data_ptr<at::BFloat16>(),
        s_out.data_ptr<float>(),
        ws0.data_ptr<float>(),
        offsets.data_ptr<int32_t>(),
        experts.data_ptr<int32_t>(),
        groups, k_total);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const int64_t total = static_cast<int64_t>(groups) * RANK * k_total;
    lora_a_grad_split_merge_kernel<<<blocks_for(total), kThreads, 0, stream>>>(
        ws0.data_ptr<float>(), nullptr,
        grad_a.data_ptr<at::BFloat16>(), nullptr,
        experts.data_ptr<int32_t>(),
        static_cast<int32_t>(splits), static_cast<int32_t>(groups),
        RANK, static_cast<int32_t>(k_total));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Not on the current best benchmark paths; retained for experimental paired gate/up LoRA-A dA runs.
void sm100_grouped_lora_a_pair_grad_bf16_cpu_right(
    const torch::Tensor& dS_gate,
    const torch::Tensor& dS_up,
    const torch::Tensor& x_cpu,
    const torch::Tensor& grad_gate,
    const torch::Tensor& grad_up,
    const torch::Tensor& offsets,
    const torch::Tensor& experts,
    int64_t list_size) {
    check_cuda_bf16_2d(dS_gate, "dS_gate");
    check_cuda_bf16_2d(dS_up, "dS_up");
    check_cpu_bf16_2d(x_cpu, "x_cpu");
    check_cuda_bf16_3d(grad_gate, "grad_gate");
    check_cuda_bf16_3d(grad_up, "grad_up");
    TORCH_CHECK(dS_gate.sizes() == dS_up.sizes(), "dS pair shape mismatch");
    TORCH_CHECK(grad_gate.sizes() == grad_up.sizes(), "grad pair shape mismatch");
    TORCH_CHECK(dS_gate.size(0) == x_cpu.size(0), "dS/source row mismatch");
    TORCH_CHECK(grad_gate.size(1) == dS_gate.size(1), "grad rank mismatch");
    TORCH_CHECK(grad_gate.size(2) == x_cpu.size(1), "grad K mismatch");
    TORCH_CHECK(dS_gate.device() == dS_up.device() && dS_gate.device() == grad_gate.device() && dS_gate.device() == grad_up.device(),
                "CUDA tensors must share device");
    TORCH_CHECK(offsets.device() == dS_gate.device() && experts.device() == dS_gate.device(),
                "metadata must be on dS CUDA device");

    const c10::cuda::CUDAGuard device_guard(dS_gate.device());
    require_sm100();
    const auto plan = validate_group_plan(offsets, experts, list_size, dS_gate.size(0), grad_gate.size(0));
    if (plan.groups <= 0 || plan.max_rows <= 0) {
        grad_gate.zero_();
        grad_up.zero_();
        return;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (use_atomic_lora_a_grad()) {
        auto grad_gate_acc = torch::zeros(grad_gate.sizes(), grad_gate.options().dtype(torch::kFloat32));
        auto grad_up_acc = torch::zeros(grad_up.sizes(), grad_up.options().dtype(torch::kFloat32));
        const int64_t max_linear = plan.max_rows * x_cpu.size(1);
        dim3 grid(blocks_for(max_linear), static_cast<unsigned int>(plan.groups));
        lora_a_grad_kernel<<<grid, kThreads, 0, stream>>>(
            dS_gate.data_ptr<at::BFloat16>(),
            dS_up.data_ptr<at::BFloat16>(),
            x_cpu.data_ptr<at::BFloat16>(),
            grad_gate_acc.data_ptr<float>(),
            grad_up_acc.data_ptr<float>(),
            offsets.data_ptr<int32_t>(),
            experts.data_ptr<int32_t>(),
            static_cast<int32_t>(plan.groups),
            static_cast<int32_t>(dS_gate.size(0)),
            static_cast<int32_t>(dS_gate.size(1)),
            static_cast<int32_t>(x_cpu.size(1)),
            max_linear);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        cast_fp32_to_bf16(grad_gate_acc, grad_gate, stream);
        cast_fp32_to_bf16(grad_up_acc, grad_up, stream);
        return;
    }
    grad_gate.zero_();
    grad_up.zero_();
    launch_lora_a_grad_tiled<64, 8, 32, 128>(
        dS_gate.data_ptr<at::BFloat16>(),
        dS_up.data_ptr<at::BFloat16>(),
        x_cpu.data_ptr<at::BFloat16>(),
        grad_gate.data_ptr<at::BFloat16>(),
        grad_up.data_ptr<at::BFloat16>(),
        offsets.data_ptr<int32_t>(),
        experts.data_ptr<int32_t>(),
        static_cast<int32_t>(plan.groups),
        static_cast<int32_t>(dS_gate.size(1)),
        static_cast<int32_t>(x_cpu.size(1)),
        stream);
}

// Not on the current best benchmark paths; LoRA-B grads there use grouped GPU/Torch paths.
void sm100_grouped_lora_b_backward_bf16_cpu_source(
    const torch::Tensor& grad_out_cpu,
    const torch::Tensor& low_rank,
    const torch::Tensor& lora_b,
    const torch::Tensor& dS,
    const torch::Tensor& grad_b,
    const torch::Tensor& offsets,
    const torch::Tensor& experts,
    int64_t list_size,
    double scale) {
    check_cpu_bf16_2d(grad_out_cpu, "grad_out_cpu");
    check_cuda_bf16_2d(low_rank, "low_rank");
    check_cuda_bf16_3d(lora_b, "lora_b");
    check_cuda_bf16_2d(dS, "dS");
    check_cuda_bf16_3d(grad_b, "grad_b");
    TORCH_CHECK(grad_out_cpu.size(0) == low_rank.size(0), "grad/low_rank row mismatch");
    TORCH_CHECK(dS.size(0) == low_rank.size(0) && dS.size(1) == low_rank.size(1), "dS shape mismatch");
    TORCH_CHECK(lora_b.size(1) == grad_out_cpu.size(1), "lora_b output dim mismatch");
    TORCH_CHECK(lora_b.size(2) == low_rank.size(1), "lora_b rank mismatch");
    TORCH_CHECK(grad_b.sizes() == lora_b.sizes(), "grad_b shape mismatch");
    TORCH_CHECK(low_rank.device() == lora_b.device() && low_rank.device() == dS.device() && low_rank.device() == grad_b.device(),
                "CUDA tensors must share device");
    TORCH_CHECK(offsets.device() == low_rank.device() && experts.device() == low_rank.device(),
                "metadata must be on low_rank CUDA device");

    const c10::cuda::CUDAGuard device_guard(low_rank.device());
    require_sm100();
    const auto plan = validate_group_plan(offsets, experts, list_size, low_rank.size(0), lora_b.size(0));
    if (plan.groups <= 0 || plan.max_rows <= 0) {
        dS.zero_();
        grad_b.zero_();
        return;
    }

    auto dS_acc = torch::zeros(dS.sizes(), dS.options().dtype(torch::kFloat32));
    auto grad_b_acc = torch::zeros(grad_b.sizes(), grad_b.options().dtype(torch::kFloat32));
    const int64_t max_linear = plan.max_rows * grad_out_cpu.size(1);
    dim3 grid(blocks_for(max_linear), static_cast<unsigned int>(plan.groups));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    lora_b_backward_kernel<<<grid, kThreads, 0, stream>>>(
        grad_out_cpu.data_ptr<at::BFloat16>(),
        low_rank.data_ptr<at::BFloat16>(),
        lora_b.data_ptr<at::BFloat16>(),
        dS_acc.data_ptr<float>(),
        grad_b_acc.data_ptr<float>(),
        offsets.data_ptr<int32_t>(),
        experts.data_ptr<int32_t>(),
        static_cast<int32_t>(plan.groups),
        static_cast<int32_t>(low_rank.size(0)),
        static_cast<int32_t>(grad_out_cpu.size(1)),
        static_cast<int32_t>(low_rank.size(1)),
        static_cast<float>(scale),
        max_linear);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    cast_fp32_to_bf16(dS_acc, dS, stream);
    cast_fp32_to_bf16(grad_b_acc, grad_b, stream);
}

}  // namespace asym_gemm::exp_act_offload
