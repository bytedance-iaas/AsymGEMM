/*
 * cpu_gemm/cpu_gemm.h — public C ABI.
 *
 * One GEMM call:  C = alpha * op(A) * op(B) + beta * C
 *
 * Layout / threading conventions match the kernels lifted from ktransformers:
 *   - A: [m, k] row-major (or [k, m] col-major)
 *   - B: typically passed as [n, k] row-major (i.e. trans_b == CG_TRANS) so
 *        the inner loop is a dot-product along k. Callers may pass
 *        CG_NO_TRANS for a [k, n] layout.
 *   - C: [m, n] row-major, dtype must be CG_F32 in v0.1.
 *
 * Quantization metadata travels in `a_scales` / `b_scales` / `a_zeros` /
 * `b_zeros`. For unquantized dtypes (F32/F16/BF16) leave them NULL and
 * `block.group_size = 0`.
 */
#ifndef CPU_GEMM_H
#define CPU_GEMM_H

#include "cpu_gemm/runtime.h"
#include "cpu_gemm/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cg_gemm_desc {
  cg_order_t       order;
  cg_trans_t       trans_a;
  cg_trans_t       trans_b;
  cg_offset_c_t    offset_c_mode;

  size_t           m, n, k;
  float            alpha;
  float            beta;

  const void*      a;
  size_t           lda;
  cg_dtype_t       dtype_a;
  cg_quant_block_t block_a;

  const void*      b;
  size_t           ldb;
  cg_dtype_t       dtype_b;
  cg_quant_block_t block_b;

  void*            c;
  size_t           ldc;
  cg_dtype_t       dtype_c;

  /* Quantization side channels. NULL if the dtype is unquantized. */
  const void*      a_scales;
  const void*      a_zeros;
  const void*      b_scales;
  const void*      b_zeros;

  /* Output offsets per the offset_c_mode (see types.h). */
  const int32_t*   output_offsets;
} cg_gemm_desc_t;

/* Synchronous, parallel GEMM. Returns CG_OK or a negative cg_status_t. */
cg_status_t cg_gemm(cg_runtime_t* rt, const cg_gemm_desc_t* desc);

/* Single-thread slice: caller owns parallelism. Splits along N into
 * (ith, nth). */
cg_status_t cg_gemm_st(const cg_gemm_desc_t* desc, int ith, int nth);

/* ----------------------------------------------------------------------
 * Offline B pre-pack for the AMX INT8 path.
 *
 * Lets callers pay the packing cost once at model-load time (e.g. into
 * pinned host memory) and reuse the packed buffer across many cg_gemm
 * calls. The dispatcher recognises desc->dtype_b == CG_INT8_PACKED_AMX
 * and skips its per-call B-pack phase.
 *
 * Padding contract: the packed buffer is sized for n padded up to
 * Int8KernelTraits::N_STEP (32); padded rows have zero data and zero
 * scale. Pre-pack does not need to know the runtime m.
 *
 * The k dimension must be a multiple of Int8KernelTraits::K_STEP (64).
 * ---------------------------------------------------------------------- */
size_t      cg_pack_b_int8_amx_size(size_t n, size_t k);
cg_status_t cg_pack_b_int8_amx(size_t n, size_t k,
                               const int8_t* b_int8,    /* [n, k] row-major */
                               const float*  b_scales,  /* [n] FP32 per-channel */
                               void*         dst);

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* CPU_GEMM_H */
