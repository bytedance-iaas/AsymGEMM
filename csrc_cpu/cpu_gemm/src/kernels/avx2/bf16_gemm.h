/*
 * AVX2 BF16 GEMM — ported from ktransformers
 *   operators/avx2/avx2_bf16_gemm.hpp  +  avx2_bf16_utils.hpp
 *
 * C[m,n] = sum_k A[m,k] * B[n,k]
 *   A: [m, k] row-major BF16
 *   B: [n, k] row-major BF16   (i.e. CG_TRANS — each B row is one output)
 *   C: [m, n] row-major FP32
 *
 * Single-threaded entry; N is sliced by (ith, nth) for parallelism.
 */
#ifndef CPU_GEMM_KERNELS_AVX2_BF16_GEMM_H
#define CPU_GEMM_KERNELS_AVX2_BF16_GEMM_H

#include <stddef.h>

#include "cpu_gemm/types.h"

namespace cpu_gemm::kernels::avx2 {

/* Compute one (ith / nth) slice of a BF16 x BF16 -> FP32 GEMM with
 * B passed in CG_TRANS layout (the inner-loop-friendly form).
 *
 * a:  [m, k] row-major BF16, row stride `lda` (in elements)
 * b:  [n, k] row-major BF16, row stride `ldb`
 * c:  [m, n] row-major FP32, row stride `ldc` (in elements)
 * alpha applied to the dot product before write; beta to existing C value.
 */
void gemm_bf16_bf16_f32(int m, int n, int k,
                        float alpha,
                        const cg_bf16_t* a, size_t lda,
                        const cg_bf16_t* b, size_t ldb,
                        float beta,
                        float* c, size_t ldc,
                        int ith, int nth);

}  // namespace cpu_gemm::kernels::avx2

#endif
